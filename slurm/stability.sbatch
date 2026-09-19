#!/bin/bash
# Stability of one dataset: outer folds for all methods (from the selection
# records, cheap) + STAB_B random half-samples for the STAB_METHODS (expensive:
# every half-sample runs a full selection).
#SBATCH --job-name=sb-stability
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=72:00:00
#SBATCH --output=slurm/logs/stability_%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

line=$(sed -n "$(( ${OFFSET:-0} + SLURM_ARRAY_TASK_ID + 1 ))p" "$TASK_LIST")
IFS=$'\t' read -r ds task <<< "$line"
read -r -a methods <<< "${STAB_METHODS:-SHAPBoost SHAPBoost-noRW GainBoost Lasso Forward-CV XGBoost}"
echo "[$(date)] $ds $task B=${STAB_B:-20} methods=${methods[*]}"
python stability.py -d "$ds" -t "$task" --subsample "${STAB_B:-20}" -m "${methods[@]}"
