"""TEST FIXTURES ONLY: tiny generated images, never seminar selection results."""

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
from jsonschema import Draft202012Validator

from group_2.cli import main
from group_2.perceptual_quality import (
    METHOD,
    METHOD_VERSION,
    score_images,
    score_manifest,
    select_subset,
)
from sensation_seminar.datasets.synthetic import SyntheticSegmentationDataset
from sensation_seminar.synthetic_data.manifest import load_manifest, write_jsonl
from sensation_seminar.synthetic_data.scoring import score_record, validate_score_overlay

REPOSITORY = Path(__file__).resolve().parents[2]
CHECKSUM = "sha256:" + "0" * 64


def _record(index, *, analysis=True, training=True):
    path = f"datasets/synthetic-data/group3/images/e1_scene_{index:04d}.png"
    key = "sha256:" + hashlib.sha256(f"group3\0{path}".encode()).hexdigest()
    return {
        "schema_version": "1.1",
        "synthetic_id": key,
        "source_group": "group3",
        "source_relpath": path,
        "original_filename": Path(path).name,
        "extension": ".png",
        "original_id": "scene",
        "real_image_relpath": "datasets/sensation_ds_v2_1_1152x768/training/images/scene.jpg",
        "real_mask_relpath": "datasets/sensation_ds_v2_1_1152x768/training/masks/scene.png",
        "real_split": "training",
        "generation_tokens": ["e1"],
        "variant": f"{index:04d}",
        "width": 4,
        "height": 4,
        "mode": "RGB",
        "content_sha256": None,
        "validity": {"status": "valid", "reason_codes": []},
        "eligibility": {"analysis": analysis, "segmentation_training": training},
        "supervision": {
            "status": "verified",
            "mask_strategy": "identity",
            "source_size": [4, 4],
            "synthetic_size": [4, 4],
            "target_size": [4, 4],
            "notes": "TEST ONLY: generated 4x4 fixture, not seminar data",
        },
        "generation_metadata": {},
    }


def _save_image(data_root, row, *, detail=False):
    path = data_root / Path(row["source_relpath"]).relative_to("datasets")
    path.parent.mkdir(parents=True, exist_ok=True)
    gray = np.array([[0, 255], [255, 0]], dtype=np.uint8) if detail else np.zeros((2, 2), dtype=np.uint8)
    rgb = np.repeat(np.tile(gray, (2, 2))[:, :, None], 3, axis=2)
    Image.fromarray(rgb).save(path)
    return path


def _score(row, value=0.5):
    return score_record(row["synthetic_id"], method=METHOD, method_version=METHOD_VERSION, score=value)


def test_perceptual_fallback_is_deterministic_and_skips_invalid():
    images = {"flat": np.zeros((3, 3)), "detail": np.eye(3), "invalid": np.array([np.nan])}
    first = score_images(images)
    assert first == score_images(images)
    assert [row["synthetic_id"] for row in first] == ["detail", "flat"]
    assert first[0]["score"] > first[1]["score"]


@pytest.mark.parametrize("invalid", [
    np.array([np.nan]), np.zeros((1, 4)), np.zeros((4, 1)),
    np.empty((0, 4)), np.full((3, 3), np.inf), np.zeros((2, 2, 0)),
    np.zeros((2, 2, 4)), np.zeros((2, 2, 2, 2)),
])
def test_invalid_arrays_have_no_score(invalid):
    assert score_images({"invalid": invalid}) == []


def test_raw_formula_direction_provenance_and_constant_pool():
    rows = score_images({"flat": np.zeros((2, 2)), "detail": np.array([[0, 255], [255, 0]])})
    by_id = {row["synthetic_id"]: row for row in rows}
    assert by_id["flat"]["score"] == 0.0
    assert by_id["detail"]["score"] == 1.0
    assert by_id["detail"]["components"]["contrast_gradient_raw"] == 637.5
    assert validate_score_overlay(rows) == rows
    assert by_id["detail"]["provenance"]["preprocessing"]["grayscale"] == "arithmetic mean of RGB channels"
    assert by_id["detail"]["direction"] == "higher_is_better"
    assert by_id["detail"]["method_version"] == "1"
    assert score_images({"only": np.zeros((2, 2))})[0]["score"] == 1.0


def test_manifest_scoring_filters_analysis_and_reports_bad_files(tmp_path):
    data_root = tmp_path / "relocated-data"
    rows = [_record(i) for i in range(5)]
    rows[1]["eligibility"]["segmentation_training"] = False  # Analysis-only still receives a score.
    rows[4]["eligibility"] = {"analysis": False, "segmentation_training": False}
    flat = _save_image(data_root, rows[0])
    _save_image(data_root, rows[1], detail=True)
    corrupt = _save_image(data_root, rows[2])
    corrupt.write_bytes(b"TEST ONLY: not an image")
    # rows[3] is missing; rows[4] is ineligible and must not be opened/reported.
    first, skipped = score_manifest(rows, data_root=data_root)
    assert (first, skipped) == score_manifest(reversed(rows), data_root=data_root)
    assert {row["synthetic_id"] for row in first} == {rows[0]["synthetic_id"], rows[1]["synthetic_id"]}
    assert {row["synthetic_id"] for row in skipped} == {rows[2]["synthetic_id"], rows[3]["synthetic_id"]}
    assert flat.read_bytes().startswith(b"\x89PNG")
    assert not (data_root / "sensation_ds_v2_1_1152x768").exists()


def test_subset_exact_budget_ties_and_training_eligibility():
    eligible = [_record(i) for i in range(550)]
    analysis_only = _record(1000, training=False)
    excluded = _record(1001, analysis=False, training=False)
    unscored = _record(1002)
    records = [*eligible, analysis_only, excluded, unscored]
    scores = [_score(row) for row in eligible] + [_score(analysis_only, 1.0), _score(excluded, 1.0)]
    subset = select_subset(records, scores, manifest_sha256=CHECKSUM)
    assert subset == select_subset(reversed(records), reversed(scores), manifest_sha256=CHECKSUM)
    assert len(subset) == 548
    assert [row["synthetic_id"] for row in subset] == sorted(row["synthetic_id"] for row in eligible)[:548]
    assert [row["rank"] for row in subset] == list(range(1, 549))
    assert all(row["selected"] is True and row["budget"] == "budget_0.25x" for row in subset)
    assert all(row["manifest_sha256"] == CHECKSUM for row in subset)


def test_subset_prefers_higher_scores_before_id_order():
    records = sorted([_record(i) for i in range(550)], key=lambda row: row["synthetic_id"])
    scores = [_score(row, 0.0) for row in records]
    scores[-1] = _score(records[-1], 1.0)
    scores[-2] = _score(records[-2], 0.75)
    subset = select_subset(records, scores, manifest_sha256=CHECKSUM)
    assert [row["synthetic_id"] for row in subset[:3]] == [
        records[-1]["synthetic_id"], records[-2]["synthetic_id"], records[0]["synthetic_id"],
    ]


def test_insufficient_valid_scores_fails_instead_of_reducing_budget():
    records = [_record(i) for i in range(548)]
    with pytest.raises(ValueError, match="Need 548.*found 547"):
        select_subset(records, [_score(row) for row in records[:-1]], manifest_sha256=CHECKSUM)


@pytest.mark.parametrize("change", [{"score": float("nan")}, {"direction": "lower_is_better"}, {"schema_version": "9"}])
def test_malformed_score_overlay_is_rejected(change):
    row = _record(0)
    score = {**_score(row), **change}
    with pytest.raises(ValueError):
        select_subset([row], [score], manifest_sha256=CHECKSUM)


def test_duplicate_ids_and_unknown_scores_fail():
    row = _record(0)
    with pytest.raises(ValueError, match="Duplicate canonical"):
        score_manifest([row, row], data_root="unused")
    with pytest.raises(ValueError, match="Duplicate score"):
        select_subset([row], [_score(row), _score(row)], manifest_sha256=CHECKSUM)
    with pytest.raises(ValueError, match="unknown synthetic_id"):
        select_subset([], [_score(row)], manifest_sha256=CHECKSUM)


def test_cli_tiny_images_roundtrip_shared_interfaces_and_repeatability(tmp_path):
    # 550 four-by-four PNGs exercise the real 548 budget, using no real dataset.
    data_root = tmp_path / "TEST-ONLY-data"
    records = [_record(i) for i in range(550)]
    schema = json.loads((REPOSITORY / "synthetic-data-seminar/manifest.schema.json").read_text())
    validator = Draft202012Validator(schema)
    for index, row in enumerate(records):
        validator.validate(row)
        _save_image(data_root, row, detail=index % 2 == 0)
    manifest = tmp_path / "TEST-ONLY-canonical.jsonl"
    write_jsonl(records, manifest)
    original_manifest = manifest.read_bytes()
    scores_path = tmp_path / "TEST-ONLY-results/scores.jsonl"
    subset_path = tmp_path / "TEST-ONLY-results/subset.jsonl"
    args = ["--data-root", str(data_root), "--manifest", str(manifest),
            "--scores-output", str(scores_path), "--subset-output", str(subset_path),
            "--overwrite"]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=os.pathsep.join([str(REPOSITORY / "src"), str(REPOSITORY / "group-2/src")]))
    result = subprocess.run([sys.executable, "-m", "group_2.cli", *args],
                            cwd=REPOSITORY, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Selected 548 records" in result.stdout
    scores_bytes, subset_bytes = scores_path.read_bytes(), subset_path.read_bytes()
    assert main(args) == 0
    assert (scores_path.read_bytes(), subset_path.read_bytes()) == (scores_bytes, subset_bytes)
    assert manifest.read_bytes() == original_manifest
    scores = [json.loads(line) for line in scores_bytes.splitlines()]
    assert len(validate_score_overlay(scores)) == 550
    joined = load_manifest(manifest, overlays=[scores_path], eligibility_filter="analysis")
    assert all(METHOD in row["scores"] for row in joined)
    subset = [json.loads(line) for line in subset_bytes.splitlines()]
    assert len({row["synthetic_id"] for row in subset}) == 548
    checksum = "sha256:" + hashlib.sha256(original_manifest).hexdigest()
    assert {row["manifest_sha256"] for row in subset} == {checksum}
    # Same ID join and dataset constructor as shared training; no model/training.
    selected_ids = {row["synthetic_id"] for row in subset}
    dataset = SyntheticSegmentationDataset(row for row in load_manifest(manifest) if row["synthetic_id"] in selected_ids)
    assert len(dataset) == 548


def test_cli_short_pool_reports_error_and_writes_no_results(tmp_path, capsys):
    manifest = tmp_path / "TEST-ONLY-canonical.jsonl"
    rows = [_record(0)]
    data_root = tmp_path / "TEST-ONLY-data"
    _save_image(data_root, rows[0])
    write_jsonl(rows, manifest)
    scores = tmp_path / "TEST-ONLY-scores.jsonl"
    subset = tmp_path / "TEST-ONLY-subset.jsonl"
    with pytest.raises(SystemExit) as error:
        main(["--manifest", str(manifest), "--data-root", str(data_root),
              "--scores-output", str(scores), "--subset-output", str(subset)])
    assert error.value.code == 2
    assert "Need 548 training-eligible records with valid scores, found 1" in capsys.readouterr().err
    assert not scores.exists() and not subset.exists()


def test_cli_refuses_to_overwrite_input_or_data(tmp_path, capsys):
    with pytest.raises(SystemExit):
        main(["--manifest", str(tmp_path / "input.jsonl"),
              "--scores-output", str(tmp_path / "input.jsonl")])
    assert "must be different paths" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["--data-root", str(tmp_path), "--scores-output", str(tmp_path / "input.png")])
    assert "protect source data" in capsys.readouterr().err


def test_cli_protects_existing_results_without_explicit_overwrite(tmp_path, capsys):
    manifest = tmp_path / "TEST-ONLY-canonical.jsonl"
    manifest.write_text("TEST ONLY: deliberately not parsed\n", encoding="utf-8")
    scores = tmp_path / "TEST-ONLY-scores.jsonl"
    scores.write_text("TEST ONLY: preserve me\n", encoding="utf-8")
    subset = tmp_path / "TEST-ONLY-subset.jsonl"

    with pytest.raises(SystemExit) as error:
        main([
            "--manifest", str(manifest),
            "--scores-output", str(scores),
            "--subset-output", str(subset),
        ])

    assert error.value.code == 2
    assert "Refusing to overwrite existing output" in capsys.readouterr().err
    assert scores.read_text(encoding="utf-8") == "TEST ONLY: preserve me\n"
    assert not subset.exists()


def test_cli_help_starts_without_reading_data():
    result = subprocess.run(
        [sys.executable, "-m", "group_2.cli", "--help"],
        cwd=REPOSITORY,
        env=dict(
            os.environ,
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONPATH=os.pathsep.join([
                str(REPOSITORY / "src"),
                str(REPOSITORY / "group-2/src"),
            ]),
        ),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--overwrite" in result.stdout


def test_quality_report_cli_writes_all_artifacts_only_to_requested_directory(tmp_path):
    data_root = tmp_path / "TEST-ONLY-data"
    records = [_record(index) for index in range(550)]
    for index, row in enumerate(records):
        _save_image(data_root, row, detail=index % 2 == 0)
    manifest = tmp_path / "TEST-ONLY-manifest.jsonl"
    scores = tmp_path / "TEST-ONLY-scores.jsonl"
    subset = tmp_path / "TEST-ONLY-subset.jsonl"
    write_jsonl(records, manifest)
    score_rows = [_score(row, index / 549) for index, row in enumerate(records)]
    write_jsonl(score_rows, scores)
    write_jsonl(select_subset(records, score_rows, manifest_sha256=CHECKSUM), subset)
    output = tmp_path / "TEST-ONLY-report"
    command = [
        sys.executable,
        str(REPOSITORY / "group-2/scripts/make_quality_report.py"),
        "--project-root", str(tmp_path),
        "--data-root", str(data_root),
        "--manifest", str(manifest),
        "--scores", str(scores),
        "--subset", str(subset),
        "--output-dir", str(output),
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    with Image.open(output / "top_bottom_examples.jpg") as image:
        assert image.size == (1600, 1060)
    with (output / "source_group_distribution.csv").open(newline="") as handle:
        assert list(csv.DictReader(handle)) == [{
            "source_group": "group3",
            "selected_count": "548",
            "selected_fraction": "1.000000",
        }]
    with (output / "source_group_eligibility_and_selection.csv").open(newline="") as handle:
        assert list(csv.DictReader(handle)) == [{
            "source_group": "group3",
            "analysis_count": "550",
            "training_candidate_count": "550",
            "selected_count": "548",
            "selection_rate": "0.996364",
        }]

    second = subprocess.run(command, capture_output=True, text=True)
    assert second.returncode != 0
    assert "Refusing to overwrite existing report artifact" in second.stderr
