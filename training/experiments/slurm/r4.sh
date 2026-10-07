#!/bin/bash
#SBATCH --job-name=ojcs-r4
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=logs/r4_%x_%A_%a.out
# Round 4 arrays. First argument: the stage (ppo | drift). Get the array size from --list:
#   python round4/r4_fold2.py --stage ppo --list       ->  sbatch --array=0-74%25   slurm/r4.sh ppo
#   python round4/r4_fold2.py --stage drift --list     ->  sbatch --array=0-1286%25 slurm/r4.sh drift
# Run each stage's --setup first (HANDOFF_ROUND4 §2). Adjust partition / time / mem / env only.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
python round4/r4_fold2.py --stage "$1" --task "${SLURM_ARRAY_TASK_ID}"
