#!/bin/bash
#SBATCH --job-name=ojcs-r5
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/r5_%x_%A_%a.out
# Round 5 arrays. First argument: the state (v2 | v1). Array size from --list:
#   python round5/r5_fold2.py --state v2 --list   ->  sbatch --array=0-N%25 slurm/r5.sh v2
# Thompson tasks need up to ~2 GB; keep --mem=8G. Adjust partition / time / env only.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python round5/r5_fold2.py --state "$1" --task "${SLURM_ARRAY_TASK_ID}"
