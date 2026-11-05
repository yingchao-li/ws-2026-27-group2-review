"""Mock-only pipeline tests plus opt-in real pyiqa integration checks."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from group_2.metric_comparison import analyze_metric_scores
from group_2.nr_iqa import METRICS, PyiqaScorer, run_metric
from group_2.nr_iqa_cli import parser as nr_parser
from sensation_seminar.synthetic_data.manifest import write_jsonl
from sensation_seminar.synthetic_data.scoring import score_record, validate_score_overlay


REPOSITORY = Path(__file__).resolve().parents[2]
MANIFEST_HASH = "sha256:" + "a" * 64


def _record(index: int, *, group: str = "group3", analysis: bool = True, training: bool = True):
    relpath = f"datasets/synthetic-data/{group}/images/fixture_{index:04d}.png"
    synthetic_id = "sha256:" + hashlib.sha256(f"{group}\0{relpath}".encode()).hexdigest()
    return {
        "synthetic_id": synthetic_id,
        "source_group": group,
        "source_relpath": relpath,
        "real_split": "training",
        "eligibility": {
            "analysis": analysis,
            "segmentation_training": training,
        },
    }


def _write_image(data_root: Path, record, *, corrupt: bool = False) -> Path:
    path = data_root / Path(record["source_relpath"]).relative_to("datasets")
    path.parent.mkdir(parents=True, exist_ok=True)
    if corrupt:
        path.write_bytes(b"TEST ONLY: corrupt image")
    else:
        Image.new("RGB", (8, 8), (31, 127, 223)).save(path)
    return path


class FakeScorer:
    def __init__(self, metric: str, *, nonfinite_filename: str | None = None):
        self.nonfinite_filename = nonfinite_filename
        self.calls: list[str] = []
        self.metadata = {
            "implementation": "mock-pyiqa",
            "implementation_version": "test-only",
            "metric_name": metric,
            "metric_mode": "NR",
            "lower_better": True,
            "score_range": [0, 100],
            "device": "cpu",
        }

    def score(self, path: Path) -> float:
        self.calls.append(path.name)
        if path.name == self.nonfinite_filename:
            return math.nan
        return float(path.stem.rsplit("_", 1)[1])


@pytest.mark.parametrize("metric", ["niqe", "brisque"])
def test_mock_metric_full_pool_normalizes_direction_and_selects_548(tmp_path, metric):
    records = [_record(index) for index in range(550)]
    data_root = tmp_path / "TEST-ONLY-data"
    for row in records:
        _write_image(data_root, row)
    output = tmp_path / f"TEST-ONLY-{metric}"
    reference = tmp_path / "TEST-ONLY-fallback-scores.jsonl"
    write_jsonl(
        [
            score_record(
                row["synthetic_id"],
                method="perceptual_fallback",
                method_version="1",
                score=index / 549,
            )
            for index, row in enumerate(records)
        ],
        reference,
    )
    summary = run_metric(
        records,
        manifest_hash=MANIFEST_HASH,
        data_root=data_root,
        output_dir=output,
        metric=metric,
        scorer=FakeScorer(metric),
        checkpoint_every=17,
        log_every=200,
        reference_scores=reference,
    )

    assert summary.complete is True
    assert (summary.processed, summary.valid, summary.skipped) == (550, 550, 0)
    scores = validate_score_overlay(
        json.loads(line) for line in (output / "scores.jsonl").read_text().splitlines()
    )
    assert len(scores) == 550
    by_raw = sorted(scores, key=lambda row: row["components"][f"{metric}_raw"])
    assert by_raw[0]["score"] == 1.0
    assert by_raw[-1]["score"] == 0.0
    assert by_raw[0]["provenance"]["raw_direction"] == "lower_is_better"
    assert by_raw[0]["provenance"]["implementation_version"] == "test-only"
    subset = [
        json.loads(line)
        for line in (output / "perceptual_quality_budget_0.25x.jsonl").read_text().splitlines()
    ]
    assert len(subset) == 548
    assert subset[0]["score_method"] == METRICS[metric].method
    assert subset[0]["method"] == METRICS[metric].subset_method
    assert [row["rank"] for row in subset] == list(range(1, 549))
    assert json.loads((output / "pool_consistency.json").read_text())["status"] == "match"
    assert json.loads((output / "completion.json").read_text())["status"] == "complete"

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_metric(
            records,
            manifest_hash=MANIFEST_HASH,
            data_root=data_root,
            output_dir=output,
            metric=metric,
            scorer=FakeScorer(metric),
            reference_scores=None,
        )


def test_checkpoint_resume_is_incremental_and_never_finalizes_partial_pool(tmp_path):
    records = [_record(index) for index in range(20)]
    data_root = tmp_path / "TEST-ONLY-data"
    for row in records:
        _write_image(data_root, row)
    output = tmp_path / "TEST-ONLY-resume"
    first_scorer = FakeScorer("niqe")
    first = run_metric(
        records,
        manifest_hash=MANIFEST_HASH,
        data_root=data_root,
        output_dir=output,
        metric="niqe",
        scorer=first_scorer,
        max_images=7,
        checkpoint_every=3,
        log_every=3,
        reference_scores=None,
    )
    assert first.complete is False and first.processed == 7
    assert len(first_scorer.calls) == 7
    assert not (output / "scores.jsonl").exists()
    assert not (output / "perceptual_quality_budget_0.25x.jsonl").exists()

    second_scorer = FakeScorer("niqe")
    second = run_metric(
        records,
        manifest_hash=MANIFEST_HASH,
        data_root=data_root,
        output_dir=output,
        metric="niqe",
        scorer=second_scorer,
        resume=True,
        max_images=5,
        checkpoint_every=2,
        log_every=2,
        reference_scores=None,
    )
    assert second.complete is False and second.processed == 12
    assert len(second_scorer.calls) == 5
    checkpoint = [json.loads(line) for line in (output / "raw_scores.checkpoint.jsonl").read_text().splitlines()]
    assert len(checkpoint) == 12
    with pytest.raises(ValueError, match="does not match.*manifest"):
        run_metric(
            records,
            manifest_hash="sha256:" + "b" * 64,
            data_root=data_root,
            output_dir=output,
            metric="niqe",
            scorer=FakeScorer("niqe"),
            resume=True,
            max_images=1,
            reference_scores=None,
        )


def test_missing_corrupt_and_nonfinite_images_are_explicit_skips(tmp_path):
    records = [_record(index) for index in range(551)]
    data_root = tmp_path / "TEST-ONLY-data"
    for index, row in enumerate(records):
        if index == 0:  # missing
            continue
        _write_image(data_root, row, corrupt=index == 1)
    output = tmp_path / "TEST-ONLY-skips"
    scorer = FakeScorer("brisque", nonfinite_filename=Path(records[2]["source_relpath"]).name)
    summary = run_metric(
        records,
        manifest_hash=MANIFEST_HASH,
        data_root=data_root,
        output_dir=output,
        metric="brisque",
        scorer=scorer,
        checkpoint_every=200,
        log_every=300,
        reference_scores=None,
    )
    assert summary.complete is True
    assert (summary.valid, summary.skipped) == (548, 3)
    skipped = [json.loads(line) for line in (output / "skipped.jsonl").read_text().splitlines()]
    assert len(skipped) == 3
    reasons = " ".join(row["reason"] for row in skipped)
    assert "FileNotFoundError" in reasons
    assert "UnidentifiedImageError" in reasons
    assert "Non-finite raw score" in reasons
    assert sum(1 for _ in (output / "perceptual_quality_budget_0.25x.jsonl").open()) == 548


def test_duplicate_manifest_ids_fail_before_scoring(tmp_path):
    row = _record(0)
    with pytest.raises(ValueError, match="Duplicate canonical"):
        run_metric(
            [row, row],
            manifest_hash=MANIFEST_HASH,
            data_root=tmp_path,
            output_dir=tmp_path / "TEST-ONLY-duplicate",
            metric="niqe",
            scorer=FakeScorer("niqe"),
            reference_scores=None,
        )


def test_cli_parsers_expose_required_batch_options():
    args = nr_parser().parse_args([
        "--metric", "niqe",
        "--device", "cpu",
        "--max-images", "4",
        "--resume",
        "--output-dir", "TEST-ONLY-output",
    ])
    assert (args.metric, args.device, args.max_images, args.resume) == ("niqe", "cpu", 4, True)
    result = subprocess.run(
        [sys.executable, "-m", "group_2.nr_iqa_cli", "--help"],
        cwd=REPOSITORY,
        env=dict(
            os.environ,
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONPATH=os.pathsep.join([str(REPOSITORY / "src"), str(REPOSITORY / "group-2/src")]),
        ),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    for option in ("--metric", "--manifest", "--data-root", "--output-dir", "--device", "--max-images", "--resume"):
        assert option in result.stdout


class _Result:
    def __init__(self, statistic: float):
        self.statistic = statistic


class FakeStatistics:
    """Dependency-free test double; production uses scipy.stats."""

    @staticmethod
    def pearsonr(x, y):
        return _Result(float(np.corrcoef(x, y)[0, 1]))

    @staticmethod
    def spearmanr(x, y):
        x_rank = np.argsort(np.argsort(x))
        y_rank = np.argsort(np.argsort(y))
        return _Result(float(np.corrcoef(x_rank, y_rank)[0, 1]))

    @staticmethod
    def kendalltau(x, y):
        concordance = 0
        pairs = 0
        for left in range(len(x)):
            products = np.sign(x[left + 1 :] - x[left]) * np.sign(y[left + 1 :] - y[left])
            concordance += int(products.sum())
            pairs += len(products)
        return _Result(concordance / pairs)


def test_metric_comparison_uses_common_ids_and_writes_statistics(tmp_path):
    records = [_record(index, group=f"group{3 + 2 * (index % 2)}") for index in range(550)]
    paths = {}
    for name, method, reverse in (
        ("fallback", "perceptual_fallback", False),
        ("niqe", "niqe_pyiqa", False),
        ("brisque", "brisque_pyiqa", True),
    ):
        path = tmp_path / f"TEST-ONLY-{name}.jsonl"
        rows = []
        for index, record in enumerate(records):
            if name == "brisque" and index == 0:
                continue
            value = (549 - index if reverse else index) / 549
            rows.append(score_record(record["synthetic_id"], method=method, method_version="1", score=value))
        write_jsonl(rows, path)
        paths[name] = path
    output = tmp_path / "TEST-ONLY-comparison"
    summary = analyze_metric_scores(
        records,
        paths,
        output_dir=output,
        manifest_sha256=MANIFEST_HASH,
        create_plots=False,
        statistics=FakeStatistics,
    )
    assert summary["global_common_valid_count"] == 549
    assert summary["coverage"]["brisque"]["missing_from_manifest_analysis_count"] == 1
    assert len(summary["correlations"]) == 3
    assert {row["common_count"] for row in summary["correlations"]} == {549}
    assert all(row["overlap_count"] in {547, 548} for row in summary["top548_overlap"])
    for name in (
        "summary.json",
        "correlations.csv",
        "top548_overlap.csv",
        "score_distributions.csv",
        "source_group_selection_rates.csv",
    ):
        assert (output / name).is_file()
    with (output / "correlations.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 3
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        analyze_metric_scores(
            records,
            paths,
            output_dir=output,
            manifest_sha256=MANIFEST_HASH,
            create_plots=False,
            statistics=FakeStatistics,
        )


@pytest.mark.integration
@pytest.mark.parametrize("metric", ["niqe", "brisque"])
def test_real_pyiqa_integration_opt_in(tmp_path, metric):
    if os.environ.get("GROUP2_RUN_PYIQA_INTEGRATION") != "1":
        pytest.skip("set GROUP2_RUN_PYIQA_INTEGRATION=1 after pre-caching pyiqa assets")
    pytest.importorskip("torch")
    pytest.importorskip("pyiqa")
    rng = np.random.default_rng(42)
    image = Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8))
    path = tmp_path / "TEST-ONLY-random.png"
    image.save(path)
    scorer = PyiqaScorer(metric, "cpu")
    assert scorer.metadata["implementation"] == "pyiqa"
    assert scorer.metadata["lower_better"] is True
    assert math.isfinite(scorer.score(path))
