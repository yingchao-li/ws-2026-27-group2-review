#!/usr/bin/env python3
"""Read-only validation of Group 2 formal scores, subset and analysis artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from PIL import Image

from group_2.perceptual_quality import METHOD, METHOD_VERSION, select_subset
from sensation_seminar.datasets.synthetic import SyntheticSegmentationDataset
from sensation_seminar.synthetic_data.manifest import load_manifest
from sensation_seminar.synthetic_data.scoring import validate_score_overlay


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("synthetic-data-seminar/manifest.jsonl"))
    parser.add_argument(
        "--manifest-schema",
        type=Path,
        default=Path("synthetic-data-seminar/manifest.schema.json"),
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=Path("group-2/results/fallback-v1/scores.jsonl"),
    )
    parser.add_argument(
        "--subset",
        type=Path,
        default=Path("group-2/results/fallback-v1/perceptual_quality_budget_0.25x.jsonl"),
    )
    parser.add_argument(
        "--distribution",
        type=Path,
        default=Path("group-2/results/fallback-v1/source_group_distribution.csv"),
    )
    parser.add_argument(
        "--eligibility-summary",
        type=Path,
        default=Path("group-2/results/fallback-v1/source_group_eligibility_and_selection.csv"),
    )
    parser.add_argument(
        "--examples",
        type=Path,
        default=Path("group-2/results/fallback-v1/top_bottom_examples.jpg"),
    )
    return parser


def _validate_distribution(path: Path, selected: Counter[str]) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    actual = {row["source_group"]: int(row["selected_count"]) for row in rows}
    if actual != dict(selected):
        raise ValueError(f"Selected source-group counts do not match {path}")
    for row in rows:
        expected = selected[row["source_group"]] / sum(selected.values())
        if not math.isclose(float(row["selected_fraction"]), expected, abs_tol=5e-7):
            raise ValueError(f"Selected source-group fraction does not match {path}")


def _validate_eligibility_summary(
    path: Path,
    manifest: list[dict[str, Any]],
    selected: Counter[str],
) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    groups = sorted({str(row["source_group"]) for row in manifest})
    if [row["source_group"] for row in rows] != groups:
        raise ValueError(f"Source groups do not match {path}")
    for summary in rows:
        group = summary["source_group"]
        analysis = sum(
            row["source_group"] == group and row["eligibility"]["analysis"]
            for row in manifest
        )
        training = sum(
            row["source_group"] == group
            and row["eligibility"]["analysis"]
            and row["eligibility"]["segmentation_training"]
            for row in manifest
        )
        chosen = selected.get(group, 0)
        expected_rate = chosen / training if training else 0.0
        observed = (
            int(summary["analysis_count"]),
            int(summary["training_candidate_count"]),
            int(summary["selected_count"]),
        )
        if observed != (analysis, training, chosen):
            raise ValueError(f"Eligibility counts for {group} do not match {path}")
        if not math.isclose(float(summary["selection_rate"]), expected_rate, abs_tol=5e-7):
            raise ValueError(f"Selection rate for {group} does not match {path}")


def validate(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_manifest(args.manifest)
    schema = json.loads(args.manifest_schema.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    for line_number, row in enumerate(manifest, 1):
        errors = sorted(validator.iter_errors(row), key=lambda error: list(error.path))
        if errors:
            raise ValueError(f"Manifest row {line_number} violates the schema: {errors[0].message}")

    manifest_ids = [str(row["synthetic_id"]) for row in manifest]
    if len(set(manifest_ids)) != len(manifest_ids):
        raise ValueError("Canonical manifest contains duplicate synthetic_id values")

    scores = validate_score_overlay(_jsonl(args.scores))
    if [row["synthetic_id"] for row in scores] != sorted(row["synthetic_id"] for row in scores):
        raise ValueError("Score rows are not in deterministic synthetic_id order")
    if any(row["method"] != METHOD or row["method_version"] != METHOD_VERSION for row in scores):
        raise ValueError(f"Expected only {METHOD} version {METHOD_VERSION} scores")

    analysis_ids = {
        row["synthetic_id"] for row in manifest if row["eligibility"]["analysis"]
    }
    score_ids = {row["synthetic_id"] for row in scores}
    if score_ids != analysis_ids:
        raise ValueError(
            "Scores must cover exactly the analysis-eligible manifest rows "
            f"(missing={len(analysis_ids - score_ids)}, extra={len(score_ids - analysis_ids)})"
        )

    try:
        raw = {
            row["synthetic_id"]: float(row["components"]["contrast_gradient_raw"])
            for row in scores
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Every score needs a numeric contrast_gradient_raw component") from error
    if not raw or not all(math.isfinite(value) for value in raw.values()):
        raise ValueError("Raw score components must be non-empty and finite")
    low, high = min(raw.values()), max(raw.values())
    for row in scores:
        value = raw[row["synthetic_id"]]
        expected = (value - low) / (high - low) if high != low else 1.0
        if not math.isclose(row["score"], expected, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"Score normalization mismatch for {row['synthetic_id']}")

    manifest_sha256 = _sha256(args.manifest)
    subset = _jsonl(args.subset)
    expected_subset = select_subset(manifest, scores, manifest_sha256=manifest_sha256)
    if subset != expected_subset:
        raise ValueError("Subset is not the deterministic eligible Top-548 reconstruction")

    selected_ids = {row["synthetic_id"] for row in subset}
    selected_manifest = [row for row in manifest if row["synthetic_id"] in selected_ids]
    if len(SyntheticSegmentationDataset(selected_manifest)) != 548:
        raise ValueError("Shared training dataset constructor did not accept exactly 548 rows")

    manifest_by_id = {row["synthetic_id"]: row for row in manifest}
    selected_groups: Counter[str] = Counter(
        str(manifest_by_id[synthetic_id]["source_group"])
        for synthetic_id in selected_ids
    )
    checked_artifacts: list[str] = []
    if args.distribution.exists():
        _validate_distribution(args.distribution, selected_groups)
        checked_artifacts.append(str(args.distribution))
    if args.eligibility_summary.exists():
        _validate_eligibility_summary(args.eligibility_summary, manifest, selected_groups)
        checked_artifacts.append(str(args.eligibility_summary))
    if args.examples.exists():
        with Image.open(args.examples) as image:
            image.verify()
        checked_artifacts.append(str(args.examples))

    return {
        "manifest": len(manifest),
        "analysis": len(analysis_ids),
        "training": sum(
            row["eligibility"]["analysis"]
            and row["eligibility"]["segmentation_training"]
            for row in manifest
        ),
        "scores": len(scores),
        "selected": len(subset),
        "manifest_sha256": manifest_sha256,
        "raw_range": [low, high],
        "normalized_range": [
            min(row["score"] for row in scores),
            max(row["score"] for row in scores),
        ],
        "selected_source_groups": dict(sorted(selected_groups.items())),
        "checked_analysis_artifacts": checked_artifacts,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = validate(args)
    print("Group 2 result validation passed")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if len(summary["checked_analysis_artifacts"]) < 3:
        print("Note: absent optional analysis artifacts were not assumed or validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
