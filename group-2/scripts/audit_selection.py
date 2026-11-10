#!/usr/bin/env python3
"""Audit original-scene, variant and source concentration without reading images."""

import argparse
import json
from pathlib import Path

from group_2.selection_audit import audit_selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("group-2/manifest.jsonl"))
    parser.add_argument("--scores", type=Path, default=Path("group-2/results/fallback-v1/scores.jsonl"))
    parser.add_argument("--subset", type=Path, default=Path("group-2/results/fallback-v1/perceptual_quality_budget_0.25x.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit_selection(args.manifest, args.scores, args.subset, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
