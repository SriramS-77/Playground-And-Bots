#!/bin/bash
#SBATCH --job-name=ojcs-scorers
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/prepare_%j.out
# Fits the three per-fold scorers ONCE into results/fold{i}_scorer/.
# Must complete before the arch-search array: every candidate has to be ranked
# against the same scorer, or TF nondeterminism changes the yardstick per task.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
python preflight.py
python exp_policy_arch_search.py --prepare
