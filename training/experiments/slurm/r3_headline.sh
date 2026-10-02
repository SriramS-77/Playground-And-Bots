#!/bin/bash
#SBATCH --job-name=ojcs-r3-headline
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=logs/r3_headline_%A_%a.out
# Only after --collect-search has written results/round3/fold2/arch_choice.json.
# Launch with the array size printed by:  python exp_round3_fold2.py --list-headline
#   e.g.  sbatch --array=0-20%21 slurm/r3_headline.sh
# Each task opens the eval block once and appends to eval_access.log.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
python exp_round3_fold2.py --headline-task "${SLURM_ARRAY_TASK_ID}"
