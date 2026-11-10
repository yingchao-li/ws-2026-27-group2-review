"""Audit tests distinguish repeated scenes, variants and unknown content duplication."""

import hashlib
import json

import pytest

from group_2.perceptual_quality import select_subset
from group_2.selection_audit import audit_selection, concentration
from sensation_seminar.synthetic_data.manifest import write_jsonl
from sensation_seminar.synthetic_data.scoring import score_record


def test_concentration_counts_scene_repetition_without_claiming_pixel_duplicates():
    rows = [dict(original_id="scene", source_group="group3", variant=v, content_sha256=None) for v in ("00", "01")]
    result = concentration(rows)
    assert result["unique_original_ids"] == 1
    assert result["images_from_repeated_originals"] == 2
    assert result["images_beyond_one_per_original"] == 1
    assert result["unique_group_original_variant_triples"] == 2
    assert result["content_duplicate_assessment"] == "incomplete_hash_coverage"
    assert concentration([])["max_images_per_original"] == 0
    for row in rows:
        row["content_sha256"] = "same-content"
    assert concentration(rows)["repeated_known_content_hashes"] == 1


@pytest.fixture
def inputs(tmp_path):
    rows = [dict(schema_version="1.1", synthetic_id=f"id-{i:04}", source_group="group3" if i % 2 else "group5",
                 original_id=f"scene-{i // 2}", variant=str(i % 2),
                 source_relpath=f"datasets/synthetic-data/group3/images/{i}.png",
                 eligibility={"analysis": True, "segmentation_training": True}) for i in range(550)]
    scores = [score_record(r["synthetic_id"], method="perceptual_fallback", method_version="1", score=i / 549) for i, r in enumerate(rows)]
    manifest, overlay, subset = (tmp_path / n for n in ("manifest.jsonl", "scores.jsonl", "subset.jsonl"))
    write_jsonl(rows, manifest)
    digest = "sha256:" + hashlib.sha256(manifest.read_bytes()).hexdigest()
    write_jsonl(scores, overlay)
    write_jsonl(select_subset(rows, scores, manifest_sha256=digest), subset)
    return manifest, overlay, subset


def test_audit_validates_and_reports_candidates_and_selected(inputs, tmp_path):
    output = tmp_path / "audit"
    before = [p.read_bytes() for p in inputs]
    result = audit_selection(*inputs, output)
    assert result["training_candidates"]["images"] == 550
    assert result["selected"]["images"] == 548
    assert result["selected"]["unique_original_ids"] == 274
    assert len(list(output.glob("*.csv"))) == 4
    assert [p.read_bytes() for p in inputs] == before
    with pytest.raises(FileExistsError):
        audit_selection(*inputs, output)


@pytest.mark.parametrize("fault", ["missing_score", "duplicate_score", "wrong_rank", "wrong_hash"])
def test_audit_rejects_incompatible_inputs(inputs, tmp_path, fault):
    manifest, scores, subset = inputs
    path = scores if "score" in fault else subset
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if fault == "missing_score":
        rows.pop()
    elif fault == "duplicate_score":
        rows.append(rows[0])
    elif fault == "wrong_rank":
        rows[0]["rank"] = 2
    else:
        rows[0]["manifest_sha256"] = "wrong"
    write_jsonl(rows, path)
    output = tmp_path / "audit"
    with pytest.raises(ValueError):
        audit_selection(manifest, scores, subset, output)
    assert not output.exists()
