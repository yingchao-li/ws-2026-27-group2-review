#!/usr/bin/env bash
#SBATCH --job-name=group2-brisque
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=group-2/logs/%x-%j.out
#SBATCH --error=group-2/logs/%x-%j.err

set -euo pipefail

: "${GROUP2_REPOSITORY:?Export GROUP2_REPOSITORY before sbatch}"
export GROUP2_METRIC=brisque
bash "$GROUP2_REPOSITORY/group-2/scripts/hpc/run_nr_iqa.sh"
