#!/bin/bash
#SBATCH --job-name=ojcs-phase0
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/phase0_%A_%a.out
# Launch with the array size printed by:
#   python exp_scorer_phase0.py --list
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
python exp_scorer_phase0.py --task "${SLURM_ARRAY_TASK_ID}"
