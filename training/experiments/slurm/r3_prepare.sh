#!/bin/bash
#SBATCH --job-name=ojcs-r3-prep
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/r3_prepare_%j.out
# Round 3, fold 2: fits the 5 cross-fitted sub-scorers + the refit ONCE into
# results/round3/fold2/scorer/, runs every gate, and the 144/144 determinism check.
# Must finish (and print "prepare: OK") before either array is launched.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
python preflight.py
python exp_round3_fold2.py --prepare
