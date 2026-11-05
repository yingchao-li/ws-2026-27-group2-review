# Group 2 NIQE/BRISQUE HPC runbook

The new metric code is implemented, but no full NIQE or BRISQUE result exists until the
commands below complete against all analysis-eligible images. This runbook never downloads
the course dataset and does not start semantic-segmentation training.

The Slurm wrappers intentionally do not set a partition or request a GPU. Supply site- and
account-specific resources to `sbatch`. Their checked-in CPU/memory/time values are only
adjustable defaults; command-line `sbatch` options override them.

## 1. Upload the Group 2 code

The shared repository, `src/`, canonical manifest and fallback scores must already exist on
HPC. From the local parent directory, replace the host/path values and upload only Group 2:

```bash
HPC_LOGIN=user@login.example.edu
HPC_REPOSITORY=/absolute/hpc/path/ws-2026-27
rsync -av --exclude '.venv*' --exclude '__pycache__' \
  ws-2026-27/group-2/ "$HPC_LOGIN:$HPC_REPOSITORY/group-2/"
```

## 2. Configure paths and install dependencies

Run on the HPC login or environment-preparation node. Reuse the site's supported Python and
PyTorch module/environment; do not copy a macOS virtual environment to Linux.

```bash
export GROUP2_REPOSITORY=/absolute/hpc/path/ws-2026-27
export GROUP2_DATA_ROOT=/absolute/hpc/path/datasets
export GROUP2_MANIFEST="$GROUP2_REPOSITORY/synthetic-data-seminar/manifest.jsonl"
export GROUP2_VENV="$GROUP2_REPOSITORY/group-2/.venv-iqa"
export GROUP2_TORCH_HOME=/absolute/hpc/path/group2-cache/torch

# Load the locally appropriate Python/PyTorch module here if the cluster requires it.
python3 -m venv --system-site-packages "$GROUP2_VENV"
source "$GROUP2_VENV/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$GROUP2_REPOSITORY/group-2/requirements-iqa.txt"

export TORCH_HOME="$GROUP2_TORCH_HOME"
export PYTHONPATH="$GROUP2_REPOSITORY/src:$GROUP2_REPOSITORY/group-2/src"
mkdir -p "$TORCH_HOME" "$GROUP2_REPOSITORY/group-2/logs"
```

`requirements-iqa.txt` pins pyiqa because the installed package version is recorded in every
score row. The official pyiqa documentation places cached assets under
`~/.cache/torch/hub/pyiqa/` by default; setting `TORCH_HOME` makes the experiment cache
explicit and reusable across jobs.

## 3. Pre-cache pyiqa metric assets

Run this before `sbatch` on a node where package/model downloads are permitted. It downloads
only pyiqa-provided metric parameters, never course images. Do not create the ready marker
unless both commands succeed.

```bash
export TORCH_HOME="$GROUP2_TORCH_HOME"
export PYTHONPATH="$GROUP2_REPOSITORY/src:$GROUP2_REPOSITORY/group-2/src"
"$GROUP2_VENV/bin/python" -m group_2.nr_iqa_cli --metric niqe --device cpu --prepare-only
"$GROUP2_VENV/bin/python" -m group_2.nr_iqa_cli --metric brisque --device cpu --prepare-only
touch "$GROUP2_REPOSITORY/group-2/.pyiqa-cache-ready"
export GROUP2_PYIQA_READY_FILE="$GROUP2_REPOSITORY/group-2/.pyiqa-cache-ready"
```

Compute jobs set Hugging Face/Transformers offline flags and refuse to start without this
marker. If pyiqa still requests an asset inside a job, stop and repeat preparation rather
than allowing an untracked runtime download.

## 4. Small smoke tests

Use separate scratch outputs. A `--max-images` run writes raw checkpoints and diagnostics
only; it deliberately does not publish normalized `scores.jsonl` or a Top-548 subset.

```bash
export PYTHONPATH="$GROUP2_REPOSITORY/src:$GROUP2_REPOSITORY/group-2/src"

"$GROUP2_VENV/bin/python" -m group_2.nr_iqa_cli \
  --metric niqe --device cpu --max-images 8 \
  --manifest "$GROUP2_MANIFEST" --data-root "$GROUP2_DATA_ROOT" \
  --output-dir "$GROUP2_REPOSITORY/group-2/results/smoke-niqe"

"$GROUP2_VENV/bin/python" -m group_2.nr_iqa_cli \
  --metric brisque --device cpu --max-images 8 \
  --manifest "$GROUP2_MANIFEST" --data-root "$GROUP2_DATA_ROOT" \
  --output-dir "$GROUP2_REPOSITORY/group-2/results/smoke-brisque"
```

Inspect `run.json`, `run.log`, `raw_scores.checkpoint.jsonl` and `skipped.jsonl` in each
smoke directory. Do not copy smoke outputs into `niqe-v1/` or `brisque-v1/`.

## 5. Submit full CPU runs

Set the partition/account values for the local FAU environment. The examples keep NIQE and
BRISQUE in different output directories and export the prepared Python/cache paths.

```bash
: "${PARTITION:?Export PARTITION with a valid Slurm partition before submission}"
export GROUP2_PYTHON="$GROUP2_VENV/bin/python"
export GROUP2_PYIQA_READY_FILE="$GROUP2_REPOSITORY/group-2/.pyiqa-cache-ready"
export TORCH_HOME="$GROUP2_TORCH_HOME"
cd "$GROUP2_REPOSITORY"

sbatch --partition="$PARTITION" --cpus-per-task=4 --mem=16G --time=12:00:00 \
  --export=ALL,GROUP2_REPOSITORY,GROUP2_DATA_ROOT,GROUP2_MANIFEST,GROUP2_PYTHON,GROUP2_PYIQA_READY_FILE,TORCH_HOME \
  "$GROUP2_REPOSITORY/group-2/scripts/hpc/score_niqe.slurm"

sbatch --partition="$PARTITION" --cpus-per-task=4 --mem=16G --time=12:00:00 \
  --export=ALL,GROUP2_REPOSITORY,GROUP2_DATA_ROOT,GROUP2_MANIFEST,GROUP2_PYTHON,GROUP2_PYIQA_READY_FILE,TORCH_HOME \
  "$GROUP2_REPOSITORY/group-2/scripts/hpc/score_brisque.slurm"
```

Do not add `--gres` unless the selected partition provides a compatible GPU. To opt in to
CUDA, add `--gres=gpu:1` to `sbatch` and export `GROUP2_DEVICE=cuda`; pyiqa direction and
device metadata will be recorded in the run.

## 6. Resume an interrupted job

Use the same metric, manifest, device, Python/pyiqa version and output directory. The CLI
rejects incompatible checkpoints.

```bash
export GROUP2_RESUME=1
sbatch --partition="$PARTITION" --cpus-per-task=4 --mem=16G --time=12:00:00 \
  --export=ALL,GROUP2_REPOSITORY,GROUP2_DATA_ROOT,GROUP2_MANIFEST,GROUP2_PYTHON,GROUP2_PYIQA_READY_FILE,TORCH_HOME,GROUP2_RESUME \
  "$GROUP2_REPOSITORY/group-2/scripts/hpc/score_niqe.slurm"

sbatch --partition="$PARTITION" --cpus-per-task=4 --mem=16G --time=12:00:00 \
  --export=ALL,GROUP2_REPOSITORY,GROUP2_DATA_ROOT,GROUP2_MANIFEST,GROUP2_PYTHON,GROUP2_PYIQA_READY_FILE,TORCH_HOME,GROUP2_RESUME \
  "$GROUP2_REPOSITORY/group-2/scripts/hpc/score_brisque.slurm"
unset GROUP2_RESUME
```

## 7. Check results and generate the comparison report

A formal directory is complete only when `completion.json`, `scores.jsonl`,
`perceptual_quality_budget_0.25x.jsonl`, `pool_consistency.json` and the checkpoint exist.
The score file should normally contain 78,840 rows and the subset exactly 548; any skipped
image must appear in `skipped.jsonl` and in the pool-consistency report.

```bash
wc -l \
  "$GROUP2_REPOSITORY/group-2/results/niqe-v1/scores.jsonl" \
  "$GROUP2_REPOSITORY/group-2/results/niqe-v1/perceptual_quality_budget_0.25x.jsonl" \
  "$GROUP2_REPOSITORY/group-2/results/brisque-v1/scores.jsonl" \
  "$GROUP2_REPOSITORY/group-2/results/brisque-v1/perceptual_quality_budget_0.25x.jsonl"

cat "$GROUP2_REPOSITORY/group-2/results/niqe-v1/completion.json"
cat "$GROUP2_REPOSITORY/group-2/results/niqe-v1/pool_consistency.json"
cat "$GROUP2_REPOSITORY/group-2/results/brisque-v1/completion.json"
cat "$GROUP2_REPOSITORY/group-2/results/brisque-v1/pool_consistency.json"

cd "$GROUP2_REPOSITORY"
PYTHONPATH=src:group-2/src "$GROUP2_VENV/bin/python" group-2/scripts/compare_metrics.py \
  --manifest "$GROUP2_MANIFEST" \
  --fallback group-2/results/fallback-v1/scores.jsonl \
  --niqe group-2/results/niqe-v1/scores.jsonl \
  --brisque group-2/results/brisque-v1/scores.jsonl \
  --output-dir group-2/results/metric-comparison-v1
```

The comparison uses normalized higher-is-better scores, reports every coverage difference,
computes correlations on the global common-ID intersection and does not claim that IQA
agreement implies downstream segmentation improvement.
