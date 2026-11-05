"""Score canonical synthetic records and select the Group 2 fallback subset."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

from sensation_seminar.synthetic_data.manifest import load_manifest, write_jsonl

from group_2.perceptual_quality import score_manifest, select_subset


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data-root", type=Path, default=Path("datasets"),
                        help="Directory containing synthetic-data/ (not the repository root)")
    result.add_argument("--manifest", type=Path,
                        default=Path("synthetic-data-seminar/manifest.jsonl"))
    result.add_argument("--scores-output", type=Path,
                        default=Path("group-2/results/fallback-v1/scores.jsonl"))
    result.add_argument("--subset-output", type=Path,
                        default=Path("group-2/results/fallback-v1/perceptual_quality_budget_0.25x.jsonl"))
    result.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing output files (disabled by default to protect formal results)",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    try:
        outputs = [args.scores_output.resolve(), args.subset_output.resolve()]
        if len(set(outputs + [args.manifest.resolve()])) != 3:
            raise ValueError("Manifest, scores output and subset output must be different paths")
        if any(path.is_relative_to(args.data_root.resolve()) for path in outputs):
            raise ValueError("Output paths must be outside --data-root to protect source data")
        existing = [path for path in (args.scores_output, args.subset_output) if path.exists()]
        if existing and not args.overwrite:
            joined = ", ".join(str(path) for path in existing)
            raise ValueError(
                f"Refusing to overwrite existing output(s): {joined}; "
                "pass --overwrite only for an intentional regeneration"
            )
        records = load_manifest(args.manifest)
        manifest_sha256 = "sha256:" + hashlib.sha256(args.manifest.read_bytes()).hexdigest()
        scores, skipped = score_manifest(records, data_root=args.data_root)
        for row in skipped:
            print(f"SKIPPED {row['source_relpath']}: {row['reason']}", file=sys.stderr)
        # Validate the full selection before writing either result file.
        subset = select_subset(records, scores, manifest_sha256=manifest_sha256)
        write_jsonl(scores, args.scores_output)
        write_jsonl(subset, args.subset_output)
    except (OSError, ValueError, KeyError) as error:
        argument_parser.error(str(error))
    print(f"Scored {len(scores)} images; skipped {len(skipped)} unreadable/invalid images")
    print(f"Scores: {args.scores_output}")
    print(f"Selected {len(subset)} records: {args.subset_output}")
    print(f"Canonical manifest: {manifest_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
