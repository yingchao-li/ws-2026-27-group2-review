"""Batch NIQE/BRISQUE scoring with safe checkpoints and full-pool normalization."""

from __future__ import annotations

import argparse
from pathlib import Path
import signal

from sensation_seminar.synthetic_data.manifest import load_manifest

from group_2.nr_iqa import (
    METRICS,
    PyiqaScorer,
    manifest_sha256,
    run_metric,
    summary_json,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--metric", required=True, choices=sorted(METRICS))
    result.add_argument(
        "--manifest",
        type=Path,
        default=Path("synthetic-data-seminar/manifest.jsonl"),
    )
    result.add_argument(
        "--data-root",
        type=Path,
        default=Path("datasets"),
        help="directory containing synthetic-data/",
    )
    result.add_argument(
        "--output-dir",
        type=Path,
        help="default: group-2/results/<metric>-v1",
    )
    result.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    result.add_argument(
        "--max-images",
        type=int,
        help="process at most this many pending images; never finalizes a partial pool",
    )
    result.add_argument(
        "--resume",
        action="store_true",
        help="continue a compatible checkpoint in --output-dir",
    )
    result.add_argument("--checkpoint-every", type=int, default=500)
    result.add_argument("--log-every", type=int, default=100)
    result.add_argument(
        "--reference-scores",
        type=Path,
        default=Path("group-2/results/fallback-v1/scores.jsonl"),
        help="fallback overlay used only for valid-pool consistency reporting",
    )
    result.add_argument(
        "--no-reference-check",
        action="store_true",
        help="skip comparison with the fallback valid-ID pool",
    )
    result.add_argument(
        "--prepare-only",
        action="store_true",
        help="instantiate pyiqa, print runtime metadata and exit without reading the manifest",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    def checkpoint_on_termination(signum, frame):
        del frame
        raise KeyboardInterrupt(f"received signal {signum}; saving checkpoint")

    signal.signal(signal.SIGTERM, checkpoint_on_termination)
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    output_dir = args.output_dir or Path(f"group-2/results/{args.metric}-v1")
    try:
        if args.max_images is not None and args.max_images <= 0:
            raise ValueError("--max-images must be positive")
        if output_dir.resolve().is_relative_to(args.data_root.resolve()):
            raise ValueError("--output-dir must be outside --data-root")
        scorer = PyiqaScorer(args.metric, args.device)
        if args.prepare_only:
            import json

            print(json.dumps(dict(scorer.metadata), indent=2, sort_keys=True))
            return 0
        records = load_manifest(args.manifest)
        summary = run_metric(
            records,
            manifest_hash=manifest_sha256(args.manifest),
            data_root=args.data_root,
            output_dir=output_dir,
            metric=args.metric,
            scorer=scorer,
            resume=args.resume,
            max_images=args.max_images,
            checkpoint_every=args.checkpoint_every,
            log_every=args.log_every,
            reference_scores=None if args.no_reference_check else args.reference_scores,
        )
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        argument_parser.error(str(error))
    print(summary_json(summary))
    if not summary.complete:
        print("Partial smoke/checkpoint run only: no normalized scores or Top-548 subset were published.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
