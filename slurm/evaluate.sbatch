#!/bin/bash
# Test-fold evaluation (incl. per-fold budget curves) of one dataset from
# slurm/tasks/datasets.tsv. Rerun after new selections: it re-evaluates all records.
#SBATCH --job-name=sb-evaluate
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/evaluate_%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh

line=$(sed -n "$(( ${OFFSET:-0} + SLURM_ARRAY_TASK_ID + 1 ))p" "$TASK_LIST")
IFS=$'\t' read -r ds task <<< "$line"
echo "[$(date)] $ds $task"
python run_evaluation.py -d "$ds" -t "$task"
python plot_curves.py -d "$ds"
