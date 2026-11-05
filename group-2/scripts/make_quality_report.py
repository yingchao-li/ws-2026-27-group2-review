#!/usr/bin/env python3
"""Generate Group 2 examples plus selected-source and eligibility summaries."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "synthetic-data-seminar/manifest.jsonl"
RESULTS_DIR = PROJECT_ROOT / "group-2/results/fallback-v1"
SCORES_PATH = RESULTS_DIR / "scores.jsonl"
SUBSET_PATH = RESULTS_DIR / "perceptual_quality_budget_0.25x.jsonl"
EXAMPLES_PATH = RESULTS_DIR / "top_bottom_examples.jpg"
DISTRIBUTION_PATH = RESULTS_DIR / "source_group_distribution.csv"
ELIGIBILITY_PATH = RESULTS_DIR / "source_group_eligibility_and_selection.csv"


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def resolve_image_path(source_relpath: str, *, project_root: Path, data_root: Path) -> Path:
    source_path = Path(source_relpath)

    if source_path.is_absolute() and source_path.is_file():
        return source_path

    project_relative = project_root / source_path
    if project_relative.is_file():
        return project_relative

    parts = source_path.parts
    if "synthetic-data" in parts:
        suffix = parts[parts.index("synthetic-data") :]
        relocated = data_root / Path(*suffix)
        if relocated.is_file():
            return relocated

    raise FileNotFoundError(f"Could not resolve image: {source_relpath}")


def load_font(size: int):
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    result.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "datasets")
    result.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    result.add_argument("--scores", type=Path, default=SCORES_PATH)
    result.add_argument("--subset", type=Path, default=SUBSET_PATH)
    result.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    result.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing report artifacts (disabled by default)",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    examples_path = args.output_dir / EXAMPLES_PATH.name
    distribution_path = args.output_dir / DISTRIBUTION_PATH.name
    eligibility_path = args.output_dir / ELIGIBILITY_PATH.name
    outputs = (examples_path, distribution_path, eligibility_path)
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing report artifact(s): {joined}; "
            "pass --overwrite only for an intentional regeneration"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scores = [
        (row["synthetic_id"], float(row["score"]))
        for row in iter_jsonl(args.scores)
    ]
    if len({synthetic_id for synthetic_id, _ in scores}) != len(scores):
        raise ValueError("Score file contains duplicate synthetic_id values")
    scores.sort(key=lambda item: (-item[1], item[0]))

    subset_rows = list(iter_jsonl(args.subset))
    subset_ids = [row["synthetic_id"] for row in subset_rows]
    if len(subset_ids) != 548 or len(set(subset_ids)) != 548:
        raise ValueError("Subset must contain exactly 548 unique synthetic_id values")

    top = scores[:10]
    bottom = list(reversed(scores[-10:]))
    wanted_ids = set(subset_ids)
    wanted_ids.update(synthetic_id for synthetic_id, _ in top + bottom)

    manifest = {}
    analysis_by_group = Counter()
    training_by_group = Counter()
    source_groups = set()
    for row in iter_jsonl(args.manifest):
        source_group = row.get("source_group", "unknown")
        source_groups.add(source_group)
        flags = row.get("eligibility", {})
        if flags.get("analysis") is True:
            analysis_by_group[source_group] += 1
        if flags.get("analysis") is True and flags.get("segmentation_training") is True:
            training_by_group[source_group] += 1
        synthetic_id = row["synthetic_id"]
        if synthetic_id in wanted_ids:
            manifest[synthetic_id] = row

    missing = wanted_ids.difference(manifest)
    if missing:
        raise RuntimeError(f"Missing {len(missing)} required IDs from manifest")

    distribution = Counter(
        manifest[synthetic_id].get("source_group", "unknown")
        for synthetic_id in subset_ids
    )

    with distribution_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_group", "selected_count", "selected_fraction"])
        for source_group, count in sorted(distribution.items()):
            writer.writerow([source_group, count, f"{count / len(subset_ids):.6f}"])

    with eligibility_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "source_group",
            "analysis_count",
            "training_candidate_count",
            "selected_count",
            "selection_rate",
        ])
        for source_group in sorted(source_groups):
            training_count = training_by_group[source_group]
            selected_count = distribution[source_group]
            selection_rate = selected_count / training_count if training_count else 0.0
            writer.writerow([
                source_group,
                analysis_by_group[source_group],
                training_count,
                selected_count,
                f"{selection_rate:.6f}",
            ])

    examples = [
        ("TOP", rank, synthetic_id, score)
        for rank, (synthetic_id, score) in enumerate(top, start=1)
    ] + [
        ("BOTTOM", rank, synthetic_id, score)
        for rank, (synthetic_id, score) in enumerate(bottom, start=1)
    ]

    columns = 5
    tile_width = 320
    image_height = 213
    label_height = 52
    tile_height = image_height + label_height
    rows = (len(examples) + columns - 1) // columns

    canvas = Image.new(
        "RGB",
        (columns * tile_width, rows * tile_height),
        color="white",
    )
    draw = ImageDraw.Draw(canvas)
    font = load_font(13)
    resampling = getattr(Image, "Resampling", Image).LANCZOS

    for index, (kind, rank, synthetic_id, score) in enumerate(examples):
        record = manifest[synthetic_id]
        image_path = resolve_image_path(
            record["source_relpath"],
            project_root=args.project_root,
            data_root=args.data_root,
        )

        with Image.open(image_path) as image:
            tile = ImageOps.fit(
                image.convert("RGB"),
                (tile_width, image_height),
                method=resampling,
            )

        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        canvas.paste(tile, (x, y))
        draw.rectangle(
            (x, y + image_height, x + tile_width, y + tile_height),
            fill="white",
        )

        source_group = record.get("source_group", "unknown")
        filename = image_path.name
        if len(filename) > 35:
            filename = filename[:32] + "..."
        label = (
            f"{kind} {rank:02d} | score={score:.4f}\n"
            f"{source_group} | {filename}"
        )
        draw.multiline_text(
            (x + 5, y + image_height + 5),
            label,
            fill="black",
            font=font,
            spacing=2,
        )

    canvas.save(examples_path, quality=92, optimize=True)

    print(f"Created: {examples_path}")
    print(f"Created: {distribution_path}")
    print(f"Created: {eligibility_path}")
    print("\nSelected 548 source-group distribution:")
    for source_group, count in sorted(distribution.items()):
        print(f"{source_group}: {count} ({count / len(subset_ids):.2%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

