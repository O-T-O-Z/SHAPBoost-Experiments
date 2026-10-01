#!/bin/bash
# Test-fold evaluation (incl. per-fold budget curves) of one dataset from
# slurm/tasks/datasets.tsv. Rerun after new selections: it re-evaluates all records.
#SBATCH --job-name=sb-evaluate
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/evaluate_%A_%a.out
set -euo pipefail
# SLURM runs a copy of this script from its spool directory, so locate the
# repository explicitly: REPO_DIR is set by submit_all.sh; for a manual sbatch it is
# derived from the path of the submitted script (slurm/<name>.sbatch).
if [[ -z "${REPO_DIR:-}" ]]; then
  script=$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/^.*[[:space:]]Command=//p' | head -n 1)
  [[ "$script" = /* ]] || script="$SLURM_SUBMIT_DIR/$script"
  REPO_DIR=$(cd "$(dirname "$script")/.." && pwd)
fi
cd "$REPO_DIR" || { echo "cannot cd to REPO_DIR=$REPO_DIR" >&2; exit 1; }
[[ -f slurm/env.sh ]] || { echo "slurm/env.sh not found in $REPO_DIR" >&2; exit 1; }
set +u  # module/conda scripts often use unset variables
source slurm/env.sh
set -u

line=$(sed -n "$(( ${OFFSET:-0} + SLURM_ARRAY_TASK_ID + 1 ))p" "$TASK_LIST")
if [[ -z "$line" ]]; then   # the task list must not be overwritten while jobs are queued
  echo "no task on line $(( ${OFFSET:-0} + SLURM_ARRAY_TASK_ID + 1 )) of $TASK_LIST" >&2
  exit 1
fi
IFS=$'\t' read -r ds task <<< "$line"
echo "[$(date)] $ds $task"
python run_evaluation.py -d "$ds" -t "$task"
python plot_curves.py -d "$ds"
