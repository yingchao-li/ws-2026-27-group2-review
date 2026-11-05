#!/usr/bin/env python3
"""Compare fallback, NIQE and optionally BRISQUE formal score overlays."""

from __future__ import annotations

import argparse
from pathlib import Path

from group_2.metric_comparison import analyze_metric_scores
from group_2.nr_iqa import manifest_sha256
from sensation_seminar.synthetic_data.manifest import load_manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, default=Path("synthetic-data-seminar/manifest.jsonl"))
    result.add_argument(
        "--fallback",
        type=Path,
        default=Path("group-2/results/fallback-v1/scores.jsonl"),
    )
    result.add_argument(
        "--niqe",
        type=Path,
        default=Path("group-2/results/niqe-v1/scores.jsonl"),
    )
    result.add_argument("--brisque", type=Path)
    result.add_argument(
        "--output-dir",
        type=Path,
        default=Path("group-2/results/metric-comparison-v1"),
    )
    result.add_argument("--overwrite", action="store_true")
    result.add_argument(
        "--no-plots",
        action="store_true",
        help="write statistical tables only (useful for dependency-light tests)",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    score_paths = {"fallback": args.fallback, "niqe": args.niqe}
    if args.brisque is not None:
        score_paths["brisque"] = args.brisque
    try:
        summary = analyze_metric_scores(
            load_manifest(args.manifest),
            score_paths,
            output_dir=args.output_dir,
            manifest_sha256=manifest_sha256(args.manifest),
            overwrite=args.overwrite,
            create_plots=not args.no_plots,
        )
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        argument_parser.error(str(error))
    print(f"Compared {', '.join(summary['metrics'])}")
    print(f"Global common valid IDs: {summary['global_common_valid_count']}")
    print(f"Outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
