"""Read-only concentration audit of the frozen fallback selection."""

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile

from group_2.perceptual_quality import select_subset
from sensation_seminar.synthetic_data.manifest import load_manifest
from sensation_seminar.synthetic_data.scoring import validate_score_overlay


def _rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def concentration(records):
    """Repeated scene IDs are not evidence of identical image content."""
    originals = Counter(row["original_id"] for row in records)
    known_hashes = Counter(row["content_sha256"] for row in records if row.get("content_sha256"))
    return {
        "images": len(records),
        "unique_original_ids": len(originals),
        "original_ids_with_multiple_images": sum(n > 1 for n in originals.values()),
        "images_from_repeated_originals": sum(n for n in originals.values() if n > 1),
        "images_beyond_one_per_original": sum(n - 1 for n in originals.values()),
        "max_images_per_original": max(originals.values(), default=0),
        "top10_original_image_share": sum(n for _, n in originals.most_common(10)) / len(records) if records else 0,
        "unique_group_original_pairs": len({(r["source_group"], r["original_id"]) for r in records}),
        "unique_group_original_variant_triples": len({
            (r["source_group"], r["original_id"], r.get("variant")) for r in records
        }),
        "images_with_content_hash": sum(known_hashes.values()),
        "repeated_known_content_hashes": sum(n > 1 for n in known_hashes.values()),
        "content_duplicate_assessment": "complete_hash_coverage" if sum(known_hashes.values()) == len(records) else "incomplete_hash_coverage",
    }


def audit_selection(manifest, scores, subset, output_dir):
    """Validate exact Top-548 before publishing a new, non-overwriting report directory."""
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite audit directory: {output}")
    records = load_manifest(manifest)
    score_rows = validate_score_overlay(_rows(scores))
    analysis_ids = {r["synthetic_id"] for r in records if r["eligibility"]["analysis"]}
    if {r["synthetic_id"] for r in score_rows} != analysis_ids:
        raise ValueError("Scores must cover exactly the analysis pool")
    selected_rows = _rows(subset)
    if selected_rows != select_subset(records, score_rows, manifest_sha256=_sha(manifest)):
        raise ValueError("Subset must equal the frozen deterministic fallback Top-548")
    candidates = [r for r in records if r["eligibility"]["analysis"] and r["eligibility"]["segmentation_training"]]
    if any(not r.get("original_id") or not r.get("source_group") for r in candidates):
        raise ValueError("Training candidates require original_id and source_group")
    selected_ids = {r["synthetic_id"] for r in selected_rows}
    selected = [r for r in candidates if r["synthetic_id"] in selected_ids]
    summary = {
        "schema_version": "1.0", "primary_method": "perceptual_fallback", "method_version": "1",
        "manifest_sha256": _sha(manifest), "scores_sha256": _sha(scores), "subset_sha256": _sha(subset),
        "analysis_images": len(analysis_ids), "training_candidates": concentration(candidates),
        "selected": concentration(selected),
        "policy": "Unique synthetic_id only; no original/variant/content deduplication or source quotas",
        "interpretation": "Descriptive concentration, not evidence of segmentation improvement or exact pixel duplication",
    }
    tables = {}
    keys = {
        "source_groups": lambda r: (r["source_group"],),
        "source_folders": lambda r: (str(PurePosixPath(r["source_relpath"]).parent),),
        "original_scenes": lambda r: (r["original_id"],),
        "variants": lambda r: (r["source_group"], str(r.get("variant") if r.get("variant") is not None else "<missing>")),
    }
    headers = {"source_groups": ["source_group"], "source_folders": ["source_folder"],
               "original_scenes": ["original_id"], "variants": ["source_group", "variant"]}
    for name, key in keys.items():
        available = Counter(key(r) for r in candidates)
        chosen = Counter(key(r) for r in selected)
        rows = []
        for value, count in sorted(available.items()):
            n = chosen[value]
            rows.append([*value, count, n, count / len(candidates), n / len(selected),
                         n / count, (n / len(selected)) / (count / len(candidates))])
        tables[name] = rows
    group_counts = Counter(r["source_group"] for r in selected)
    summary["selected_source_group_hhi"] = sum((n / len(selected)) ** 2 for n in group_counts.values())
    summary["selected_largest_source_group_share"] = max(group_counts.values()) / len(selected)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".selection-audit-", dir=output.parent) as temporary:
        staging = Path(temporary) / "report"
        staging.mkdir()
        (staging / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for name, rows in tables.items():
            with (staging / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(headers[name] + ["training_candidates", "selected", "candidate_share", "selected_share", "selection_rate", "representation_ratio"])
                writer.writerows(rows)
        staging.rename(output)
    return summary
