#!/bin/bash
#SBATCH --job-name=ojcs-r3-search
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=logs/r3_search_%A_%a.out
# Launch with the array size printed by:  python exp_round3_fold2.py --list-search
#   e.g.  sbatch --array=0-224%25 slurm/r3_search.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
python exp_round3_fold2.py --search-task "${SLURM_ARRAY_TASK_ID}"
