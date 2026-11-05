"""Contrast-gradient fallback scores and deterministic fixed-budget selection."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image

from sensation_seminar.synthetic_data.scoring import (
    normalize_scores,
    score_record,
    validate_score_overlay,
)
from sensation_seminar.synthetic_data.selection import budget_size

METHOD = "perceptual_fallback"
METHOD_VERSION = "1"
SUBSET_METHOD = "perceptual_quality_budget_0.25x"


def _raw_score(image: np.ndarray) -> float | None:
    """Keep the shipped formula; reject images without two usable spatial axes."""
    try:
        array = np.asarray(image, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if array.ndim not in (2, 3) or min(array.shape[:2]) < 2:
        return None
    if array.ndim == 3 and array.shape[2] != 3:
        return None
    if not np.isfinite(array).all():
        return None
    gray = array.mean(axis=2) if array.ndim == 3 else array
    raw = float(
        gray.std()
        + np.abs(np.diff(gray, axis=0)).mean()
        + np.abs(np.diff(gray, axis=1)).mean()
    )
    return raw if math.isfinite(raw) else None


def _score_rows(raw: Mapping[str, float]) -> list[dict[str, Any]]:
    scores = normalize_scores(raw, higher_is_better=True)
    return validate_score_overlay([
        score_record(
            key,
            method=METHOD,
            method_version=METHOD_VERSION,
            score=scores[key],
            components={"contrast_gradient_raw": raw[key]},
            provenance={
                "metric": "contrast_and_gradient",
                "raw_direction": "higher_is_better",
                "preprocessing": {
                    "file_decode": "Pillow convert('RGB'), uint8 0..255",
                    "array_input": "RGB or grayscale, float64; no intensity rescaling",
                    "grayscale": "arithmetic mean of RGB channels",
                    "geometry": "native size; no resize, crop or EXIF rotation",
                },
                "normalization": "min-max over all successfully scored analysis rows; constant pool = 1",
            },
        )
        for key in sorted(scores)
    ])


def score_images(images: Mapping[str, np.ndarray]) -> list[dict]:
    """Score small in-memory fixtures; production uses one image at a time below."""
    raw = {}
    for key, image in images.items():
        value = _raw_score(image)
        if value is not None:
            raw[key] = value
    return _score_rows(raw)


def _index_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    indexed = {}
    for record in records:
        key = record["synthetic_id"]
        if key in indexed:
            raise ValueError(f"Duplicate canonical synthetic_id: {key}")
        indexed[key] = record
    return indexed


def _image_path(source_relpath: str, data_root: Path) -> Path:
    """Relocate the canonical synthetic-data suffix without changing its ID."""
    parts = Path(source_relpath).parts
    if "synthetic-data" not in parts:
        raise ValueError(f"source_relpath must contain synthetic-data/: {source_relpath}")
    suffix = parts[parts.index("synthetic-data"):]
    if ".." in suffix:
        raise ValueError(f"Invalid source_relpath: {source_relpath}")
    return data_root.joinpath(*suffix)


def score_manifest(
    records: Iterable[Mapping[str, Any]], *, data_root: str | Path
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Score analysis rows only, returning diagnostics for unreadable images."""
    indexed = _index_records(records)
    raw = {}
    skipped = []
    for key, record in sorted(indexed.items()):
        if record.get("eligibility", {}).get("analysis") is not True:
            continue
        source_relpath = record["source_relpath"]
        path = _image_path(source_relpath, Path(data_root))
        try:
            with Image.open(path) as source:
                value = _raw_score(np.asarray(source.convert("RGB")))
            if value is None:
                raise ValueError("Image needs at least 2x2 finite pixels")
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            skipped.append({
                "synthetic_id": key,
                "source_relpath": source_relpath,
                "reason": f"{type(error).__name__}: {error}",
            })
            continue
        raw[key] = value
    return _score_rows(raw), skipped


def select_subset(
    records: Iterable[Mapping[str, Any]],
    scores: Iterable[Mapping[str, Any]],
    *,
    manifest_sha256: str,
) -> list[dict[str, Any]]:
    """Return the unchanged fallback Top-548 selection."""
    return select_ranked_subset(
        records,
        scores,
        manifest_sha256=manifest_sha256,
        score_method=METHOD,
        score_method_version=METHOD_VERSION,
        subset_method=SUBSET_METHOD,
    )


def select_ranked_subset(
    records: Iterable[Mapping[str, Any]],
    scores: Iterable[Mapping[str, Any]],
    *,
    manifest_sha256: str,
    score_method: str,
    score_method_version: str,
    subset_method: str,
) -> list[dict[str, Any]]:
    """Return exactly 548 scored training candidates, ordered by (-score, ID)."""
    indexed = _index_records(records)
    validated = validate_score_overlay(scores)
    candidates = []
    for score in validated:
        key = score["synthetic_id"]
        if key not in indexed:
            raise ValueError(f"Score references unknown synthetic_id: {key}")
        if score["method"] != score_method or score["method_version"] != score_method_version:
            raise ValueError(f"Expected {score_method} version {score_method_version} scores")
        flags = indexed[key].get("eligibility", {})
        if flags.get("analysis") is True and flags.get("segmentation_training") is True:
            candidates.append(score)
    size = budget_size("0.25x")
    if len(candidates) < size:
        raise ValueError(
            f"Need {size} training-eligible records with valid scores, found {len(candidates)}"
        )
    candidates.sort(key=lambda row: (-row["score"], row["synthetic_id"]))
    return [
        {
            "schema_version": "1.0",
            "synthetic_id": row["synthetic_id"],
            "selected": True,
            "rank": rank,
            "method": subset_method,
            "budget": "budget_0.25x",
            "seed": 42,
            "source_group": indexed[row["synthetic_id"]]["source_group"],
            "score": row["score"],
            "score_method": score_method,
            "score_method_version": score_method_version,
            "manifest_sha256": manifest_sha256,
        }
        for rank, row in enumerate(candidates[:size], 1)
    ]
