#!/bin/bash
#SBATCH --job-name=ojcs-arch
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/arch_%A_%a.out
# Launch with the array size printed by:
#   python exp_policy_arch_search.py --list --seeds 5
# e.g.  sbatch --array=0-224%25 slurm/arch_search.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
python exp_policy_arch_search.py --task "${SLURM_ARRAY_TASK_ID}" \
       --episodes 200 --eval-seeds 20
