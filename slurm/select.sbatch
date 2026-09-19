#!/bin/bash
# One feature selection = one (dataset, method, split) line of $TASK_LIST.
# Single CPU on purpose: the recorded cpu_seconds are then uncontended and can be
# used for the runtime comparison (review point 4).
#SBATCH --job-name=sb-select
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/select_%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

line=$(sed -n "$(( ${OFFSET:-0} + SLURM_ARRAY_TASK_ID + 1 ))p" "$TASK_LIST")
IFS=$'\t' read -r ds task method split <<< "$line"
echo "[$(date)] $ds $task $method $split"
python run_selection.py -d "$ds" -t "$task" -m "$method" --splits "$split"
