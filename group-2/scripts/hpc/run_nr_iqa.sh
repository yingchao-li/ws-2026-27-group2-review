#!/usr/bin/env bash
# Common runtime used by the NIQE and BRISQUE Slurm wrappers.

set -euo pipefail

: "${GROUP2_REPOSITORY:?Set GROUP2_REPOSITORY to the ws-2026-27 checkout}"
: "${GROUP2_DATA_ROOT:?Set GROUP2_DATA_ROOT to the directory containing synthetic-data/}"
: "${GROUP2_METRIC:?Set GROUP2_METRIC to niqe or brisque}"
: "${GROUP2_PYIQA_READY_FILE:?Set GROUP2_PYIQA_READY_FILE after pre-caching pyiqa assets}"
: "${GROUP2_MANIFEST:=$GROUP2_REPOSITORY/synthetic-data-seminar/manifest.jsonl}"
: "${GROUP2_OUTPUT_DIR:=$GROUP2_REPOSITORY/group-2/results/$GROUP2_METRIC-v1}"
: "${GROUP2_PYTHON:=python}"
: "${GROUP2_DEVICE:=cpu}"
: "${GROUP2_CHECKPOINT_EVERY:=500}"
: "${GROUP2_LOG_EVERY:=100}"
: "${GROUP2_RESUME:=0}"

if [[ "$GROUP2_METRIC" != "niqe" && "$GROUP2_METRIC" != "brisque" ]]; then
  echo "GROUP2_METRIC must be niqe or brisque" >&2
  exit 2
fi
if [[ ! -f "$GROUP2_PYIQA_READY_FILE" ]]; then
  echo "pyiqa cache marker missing: $GROUP2_PYIQA_READY_FILE" >&2
  echo "Run both --prepare-only commands on a preparation node first." >&2
  exit 2
fi
if [[ "$GROUP2_OUTPUT_DIR" == *"fallback-v1"* ]]; then
  echo "Refusing to write NIQE/BRISQUE results into fallback-v1" >&2
  exit 2
fi

cd "$GROUP2_REPOSITORY"
export PYTHONPATH="$GROUP2_REPOSITORY/src:$GROUP2_REPOSITORY/group-2/src"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

command=(
  "$GROUP2_PYTHON" -m group_2.nr_iqa_cli
  --metric "$GROUP2_METRIC"
  --manifest "$GROUP2_MANIFEST"
  --data-root "$GROUP2_DATA_ROOT"
  --output-dir "$GROUP2_OUTPUT_DIR"
  --device "$GROUP2_DEVICE"
  --checkpoint-every "$GROUP2_CHECKPOINT_EVERY"
  --log-every "$GROUP2_LOG_EVERY"
  --reference-scores "$GROUP2_REPOSITORY/group-2/results/fallback-v1/scores.jsonl"
)

if [[ "$GROUP2_RESUME" == "1" ]]; then
  command+=(--resume)
fi
if [[ -n "${GROUP2_MAX_IMAGES:-}" ]]; then
  command+=(--max-images "$GROUP2_MAX_IMAGES")
fi

echo "metric=$GROUP2_METRIC device=$GROUP2_DEVICE output=$GROUP2_OUTPUT_DIR resume=$GROUP2_RESUME"
"${command[@]}"
