"""Resumable pyiqa NIQE/BRISQUE scoring over the canonical manifest."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib import import_module, metadata as importlib_metadata
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Protocol

from PIL import Image

from sensation_seminar.synthetic_data.manifest import write_jsonl
from sensation_seminar.synthetic_data.scoring import (
    normalize_scores,
    score_record,
    validate_score_overlay,
)

from group_2.perceptual_quality import (
    _image_path,
    _index_records,
    select_ranked_subset,
)


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    method: str
    method_version: str
    subset_method: str


METRICS = {
    "niqe": MetricDefinition(
        name="niqe",
        method="niqe_pyiqa",
        method_version="1",
        subset_method="perceptual_quality_niqe_budget_0.25x",
    ),
    "brisque": MetricDefinition(
        name="brisque",
        method="brisque_pyiqa",
        method_version="1",
        subset_method="perceptual_quality_brisque_budget_0.25x",
    ),
}


class RawScorer(Protocol):
    metadata: Mapping[str, Any]

    def score(self, path: Path) -> float:
        """Return one raw metric score."""


class PyiqaScorer:
    """Small lazy adapter around the official pyiqa inference API."""

    def __init__(self, metric: str, device: str):
        if metric not in METRICS:
            raise ValueError(f"Unsupported metric: {metric}")
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")

        try:
            torch = import_module("torch")
            pyiqa = import_module("pyiqa")
        except ImportError as error:
            raise RuntimeError(
                "NIQE/BRISQUE require torch and pyiqa; install group-2/requirements-iqa.txt"
            ) from error
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")

        available = set(pyiqa.list_models())
        if metric not in available:
            raise RuntimeError(f"Installed pyiqa does not provide metric {metric!r}")
        model = pyiqa.create_metric(metric, device=device)
        if not hasattr(model, "lower_better"):
            raise RuntimeError("pyiqa metric does not expose required lower_better metadata")

        try:
            package_version = importlib_metadata.version("pyiqa")
        except importlib_metadata.PackageNotFoundError:
            package_version = str(getattr(pyiqa, "__version__", "unknown"))
        score_range = getattr(model, "score_range", None)
        if isinstance(score_range, tuple):
            score_range = list(score_range)

        self._model = model
        self._torch = torch
        self.metadata = {
            "implementation": "pyiqa",
            "implementation_version": package_version,
            "metric_name": metric,
            "metric_mode": str(getattr(model, "metric_mode", "NR")),
            "lower_better": bool(model.lower_better),
            "score_range": score_range,
            "device": device,
        }

    def score(self, path: Path) -> float:
        with self._torch.no_grad():
            result = self._model(str(path))
        if hasattr(result, "detach"):
            result = result.detach()
        if hasattr(result, "cpu"):
            result = result.cpu()
        if hasattr(result, "item"):
            result = result.item()
        return float(result)


@dataclass(frozen=True)
class RunPaths:
    output_dir: Path
    checkpoint: Path
    skipped: Path
    scores: Path
    subset: Path
    pool_consistency: Path
    run_metadata: Path
    completion: Path
    log: Path

    @classmethod
    def for_directory(cls, output_dir: str | Path) -> "RunPaths":
        root = Path(output_dir)
        return cls(
            output_dir=root,
            checkpoint=root / "raw_scores.checkpoint.jsonl",
            skipped=root / "skipped.jsonl",
            scores=root / "scores.jsonl",
            subset=root / "perceptual_quality_budget_0.25x.jsonl",
            pool_consistency=root / "pool_consistency.json",
            run_metadata=root / "run.json",
            completion=root / "completion.json",
            log=root / "run.log",
        )


@dataclass(frozen=True)
class RunSummary:
    metric: str
    expected_analysis: int
    processed: int
    valid: int
    skipped: int
    complete: bool
    output_dir: str


def manifest_sha256(path: str | Path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _log(paths: RunPaths, message: str) -> None:
    line = f"{_utc_now()} {message}"
    print(line, flush=True)
    with paths.log.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _checkpoint_rows(states: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(states[key]) for key in sorted(states)]


def _write_checkpoint(states: Mapping[str, Mapping[str, Any]], paths: RunPaths) -> None:
    write_jsonl(_checkpoint_rows(states), paths.checkpoint)
    skipped = [row for row in _checkpoint_rows(states) if row["status"] == "skipped"]
    write_jsonl(skipped, paths.skipped)


def _load_checkpoint(
    path: Path,
    *,
    metric: str,
    expected_manifest_sha256: str,
) -> dict[str, dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid checkpoint JSON on line {line_number}") from error
            required = {
                "schema_version",
                "metric",
                "manifest_sha256",
                "synthetic_id",
                "source_relpath",
                "status",
            }
            missing = required - row.keys()
            if missing:
                raise ValueError(f"Checkpoint line {line_number} is missing {sorted(missing)}")
            if row["schema_version"] != "1.0" or row["metric"] != metric:
                raise ValueError("Checkpoint schema or metric does not match this run")
            if row["manifest_sha256"] != expected_manifest_sha256:
                raise ValueError("Checkpoint was created from a different manifest")
            if row["status"] not in {"ok", "skipped"}:
                raise ValueError(f"Invalid checkpoint status on line {line_number}")
            if row["status"] == "ok":
                value = float(row.get("raw_score", math.nan))
                if not math.isfinite(value):
                    raise ValueError(f"Invalid checkpoint raw score on line {line_number}")
            elif not row.get("reason"):
                raise ValueError(f"Skipped checkpoint line {line_number} needs a reason")
            key = str(row["synthetic_id"])
            if key in states:
                raise ValueError(f"Duplicate checkpoint synthetic_id: {key}")
            states[key] = dict(row)
    return states


def _validate_resume_metadata(
    path: Path,
    *,
    metric: str,
    expected_manifest_sha256: str,
    runtime: Mapping[str, Any],
) -> None:
    stored = json.loads(path.read_text(encoding="utf-8"))
    if stored.get("metric") != metric or stored.get("manifest_sha256") != expected_manifest_sha256:
        raise ValueError("Run metadata does not match the requested metric and manifest")
    for key in (
        "implementation",
        "implementation_version",
        "metric_name",
        "lower_better",
        "device",
    ):
        if stored.get("runtime", {}).get(key) != runtime.get(key):
            raise ValueError(f"Runtime metadata changed since checkpoint creation: {key}")


def _pool_consistency(
    valid_ids: set[str], reference_scores: Path | None
) -> dict[str, Any]:
    if reference_scores is None:
        return {"status": "not_requested", "valid_metric_ids": len(valid_ids)}
    if not reference_scores.exists():
        raise FileNotFoundError(
            f"Fallback reference score overlay does not exist: {reference_scores}; "
            "provide the formal path or explicitly use --no-reference-check"
        )
    with reference_scores.open(encoding="utf-8") as handle:
        reference = validate_score_overlay(
            json.loads(line) for line in handle if line.strip()
        )
    reference_ids = {str(row["synthetic_id"]) for row in reference}
    missing = sorted(reference_ids - valid_ids)
    extra = sorted(valid_ids - reference_ids)
    return {
        "status": "match" if not missing and not extra else "different",
        "reference_scores": str(reference_scores),
        "reference_ids": len(reference_ids),
        "valid_metric_ids": len(valid_ids),
        "missing_from_metric_count": len(missing),
        "extra_in_metric_count": len(extra),
        "missing_from_metric_ids": missing,
        "extra_in_metric_ids": extra,
    }


def _score_rows(
    raw: Mapping[str, float],
    definition: MetricDefinition,
    runtime: Mapping[str, Any],
    *,
    manifest_hash: str,
) -> list[dict[str, Any]]:
    lower_better = bool(runtime["lower_better"])
    normalized = normalize_scores(raw, higher_is_better=not lower_better)
    rows = [
        score_record(
            synthetic_id,
            method=definition.method,
            method_version=definition.method_version,
            score=normalized[synthetic_id],
            components={f"{definition.name}_raw": raw[synthetic_id]},
            provenance={
                "metric": definition.name,
                "raw_direction": "lower_is_better" if lower_better else "higher_is_better",
                "implementation": runtime["implementation"],
                "implementation_version": runtime["implementation_version"],
                "implementation_metadata": dict(runtime),
                "input": "decoded image path handled by pyiqa; one image per inference call",
                "normalization": (
                    "min-max over all successfully scored analysis rows from the exact manifest; "
                    "direction converted using runtime lower_better metadata; constant pool = 1"
                ),
                "manifest_sha256": manifest_hash,
            },
        )
        for synthetic_id in sorted(normalized)
    ]
    return validate_score_overlay(rows)


def _publish_outputs(
    scores: list[dict[str, Any]],
    subset: list[dict[str, Any]],
    paths: RunPaths,
) -> None:
    score_pending = paths.scores.with_suffix(paths.scores.suffix + ".pending")
    subset_pending = paths.subset.with_suffix(paths.subset.suffix + ".pending")
    write_jsonl(scores, score_pending)
    write_jsonl(subset, subset_pending)
    score_pending.replace(paths.scores)
    subset_pending.replace(paths.subset)


def run_metric(
    records: Iterable[Mapping[str, Any]],
    *,
    manifest_hash: str,
    data_root: str | Path,
    output_dir: str | Path,
    metric: str,
    scorer: RawScorer,
    resume: bool = False,
    max_images: int | None = None,
    checkpoint_every: int = 500,
    log_every: int = 100,
    reference_scores: str | Path | None = None,
) -> RunSummary:
    """Score a bounded batch or complete run, normalizing only after full coverage."""
    if metric not in METRICS:
        raise ValueError(f"Unsupported metric: {metric}")
    if max_images is not None and max_images <= 0:
        raise ValueError("max_images must be positive")
    if checkpoint_every <= 0 or log_every <= 0:
        raise ValueError("checkpoint_every and log_every must be positive")

    runtime = dict(scorer.metadata)
    required_runtime = {
        "implementation",
        "implementation_version",
        "metric_name",
        "lower_better",
        "device",
    }
    missing_runtime = required_runtime - runtime.keys()
    if missing_runtime:
        raise ValueError(f"Scorer metadata is missing {sorted(missing_runtime)}")
    if runtime["metric_name"] != metric:
        raise ValueError("Scorer metric metadata does not match requested metric")

    definition = METRICS[metric]
    paths = RunPaths.for_directory(output_dir)
    indexed = _index_records(records)
    analysis = {
        key: row
        for key, row in indexed.items()
        if row.get("eligibility", {}).get("analysis") is True
    }
    expected_ids = set(analysis)
    managed = (
        paths.checkpoint,
        paths.skipped,
        paths.scores,
        paths.subset,
        paths.pool_consistency,
        paths.run_metadata,
        paths.completion,
        paths.log,
    )

    if resume:
        if paths.completion.exists():
            completion = json.loads(paths.completion.read_text(encoding="utf-8"))
            if completion.get("metric") != metric or completion.get("manifest_sha256") != manifest_hash:
                raise ValueError("Existing completion marker belongs to another run")
            missing_outputs = [
                str(path)
                for path in (paths.scores, paths.subset, paths.pool_consistency, paths.checkpoint)
                if not path.exists()
            ]
            if missing_outputs:
                raise ValueError(
                    "Completion marker exists but required outputs are missing: "
                    + ", ".join(missing_outputs)
                )
            return RunSummary(
                metric=metric,
                expected_analysis=int(completion["expected_analysis"]),
                processed=int(completion["processed"]),
                valid=int(completion["valid"]),
                skipped=int(completion["skipped"]),
                complete=True,
                output_dir=str(paths.output_dir),
            )
        if not paths.checkpoint.exists() or not paths.run_metadata.exists():
            raise ValueError("--resume requires run.json and raw_scores.checkpoint.jsonl")
        _validate_resume_metadata(
            paths.run_metadata,
            metric=metric,
            expected_manifest_sha256=manifest_hash,
            runtime=runtime,
        )
        states = _load_checkpoint(
            paths.checkpoint,
            metric=metric,
            expected_manifest_sha256=manifest_hash,
        )
        unknown = set(states) - expected_ids
        if unknown:
            raise ValueError(f"Checkpoint contains {len(unknown)} IDs outside the analysis pool")
    else:
        existing = [str(path) for path in managed if path.exists()]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite an existing metric run: " + ", ".join(existing)
            )
        paths.output_dir.mkdir(parents=True, exist_ok=True)
        states = {}

    pending = sorted(expected_ids - set(states))
    if max_images is not None:
        pending = pending[:max_images]
    started = time.monotonic()
    initial_count = len(states)
    run_metadata = {
        "schema_version": "1.0",
        "status": "running",
        "metric": metric,
        "method": definition.method,
        "method_version": definition.method_version,
        "manifest_sha256": manifest_hash,
        "data_root": str(Path(data_root)),
        "reference_scores": str(reference_scores) if reference_scores is not None else None,
        "expected_analysis": len(expected_ids),
        "runtime": runtime,
        "normalization_scope": "full successfully scored analysis pool only",
        "updated_at": _utc_now(),
    }
    _write_json_atomic(run_metadata, paths.run_metadata)
    _log(
        paths,
        f"metric={metric} resume={resume} expected={len(expected_ids)} "
        f"already_processed={initial_count} scheduled={len(pending)}",
    )

    since_checkpoint = 0
    try:
        for batch_index, synthetic_id in enumerate(pending, 1):
            record = analysis[synthetic_id]
            source_relpath = str(record["source_relpath"])
            checkpoint_row: dict[str, Any] = {
                "schema_version": "1.0",
                "metric": metric,
                "manifest_sha256": manifest_hash,
                "synthetic_id": synthetic_id,
                "source_relpath": source_relpath,
            }
            try:
                path = _image_path(source_relpath, Path(data_root))
                with Image.open(path) as image:
                    image.verify()
                raw_score = float(scorer.score(path))
                if not math.isfinite(raw_score):
                    raise ValueError(f"Non-finite raw score: {raw_score}")
                checkpoint_row.update({"status": "ok", "raw_score": raw_score})
            except Exception as error:  # Per-image failures are explicit checkpoint rows.
                checkpoint_row.update(
                    {
                        "status": "skipped",
                        "reason": f"{type(error).__name__}: {error}",
                    }
                )
            states[synthetic_id] = checkpoint_row
            since_checkpoint += 1
            if since_checkpoint >= checkpoint_every:
                _write_checkpoint(states, paths)
                since_checkpoint = 0
            if batch_index % log_every == 0 or batch_index == len(pending):
                elapsed = time.monotonic() - started
                valid_now = sum(row["status"] == "ok" for row in states.values())
                skipped_now = len(states) - valid_now
                _log(
                    paths,
                    f"progress={len(states)}/{len(expected_ids)} valid={valid_now} "
                    f"skipped={skipped_now} elapsed_seconds={elapsed:.1f}",
                )
    finally:
        _write_checkpoint(states, paths)

    valid_count = sum(row["status"] == "ok" for row in states.values())
    skipped_count = len(states) - valid_count
    complete = set(states) == expected_ids
    run_metadata.update(
        {
            "status": "scanned" if complete else "partial",
            "processed": len(states),
            "valid": valid_count,
            "skipped": skipped_count,
            "updated_at": _utc_now(),
        }
    )
    _write_json_atomic(run_metadata, paths.run_metadata)

    if not complete:
        _log(
            paths,
            "partial run: raw checkpoint saved; normalization and Top-548 were not produced",
        )
        return RunSummary(
            metric=metric,
            expected_analysis=len(expected_ids),
            processed=len(states),
            valid=valid_count,
            skipped=skipped_count,
            complete=False,
            output_dir=str(paths.output_dir),
        )

    raw = {
        synthetic_id: float(row["raw_score"])
        for synthetic_id, row in states.items()
        if row["status"] == "ok"
    }
    scores = _score_rows(
        raw,
        definition,
        runtime,
        manifest_hash=manifest_hash,
    )
    subset = select_ranked_subset(
        indexed.values(),
        scores,
        manifest_sha256=manifest_hash,
        score_method=definition.method,
        score_method_version=definition.method_version,
        subset_method=definition.subset_method,
    )
    pool = _pool_consistency(
        set(raw),
        Path(reference_scores) if reference_scores is not None else None,
    )
    _write_json_atomic(pool, paths.pool_consistency)
    _publish_outputs(scores, subset, paths)

    completion = {
        "schema_version": "1.0",
        "status": "complete",
        "metric": metric,
        "method": definition.method,
        "method_version": definition.method_version,
        "manifest_sha256": manifest_hash,
        "expected_analysis": len(expected_ids),
        "processed": len(states),
        "valid": valid_count,
        "skipped": skipped_count,
        "scores": str(paths.scores),
        "subset": str(paths.subset),
        "completed_at": _utc_now(),
    }
    _write_json_atomic(completion, paths.completion)
    run_metadata.update({"status": "complete", "completed_at": completion["completed_at"]})
    _write_json_atomic(run_metadata, paths.run_metadata)
    _log(
        paths,
        f"complete scores={len(scores)} selected={len(subset)} pool_status={pool['status']}",
    )
    return RunSummary(
        metric=metric,
        expected_analysis=len(expected_ids),
        processed=len(states),
        valid=valid_count,
        skipped=skipped_count,
        complete=True,
        output_dir=str(paths.output_dir),
    )


def summary_json(summary: RunSummary) -> str:
    return json.dumps(asdict(summary), indent=2, sort_keys=True)
