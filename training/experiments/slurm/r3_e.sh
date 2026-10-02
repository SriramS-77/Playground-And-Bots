#!/bin/bash
#SBATCH --job-name=ojcs-r3-e
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=logs/r3_%x_%A_%a.out
# Round 3 experiment arrays. First argument: the experiment id (e2 e3 e4 e6 e7 e8 e9).
# Get the array size from its --list, e.g.
#   python round3/e7_proxy_reward.py --list
#   sbatch --array=0-92%25 slurm/r3_e.sh e7
# Only after --collect-search has written results/round3/fold2/arch_choice.json; E3's
# landscape/abandonment, E6's bootstrap and E8's sweep also need the headline's agents/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
EXP="$1"
SCRIPT=$(ls round3/${EXP}_*.py | head -1)
python "$SCRIPT" --task "${SLURM_ARRAY_TASK_ID}"
