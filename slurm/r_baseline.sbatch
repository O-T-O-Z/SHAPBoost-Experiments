#!/bin/bash
# C-index boosting + stability selection [B] for one (dataset, split) line of
# $TASK_LIST. The 100 stability-selection refits run on the job's cores.
#SBATCH --job-name=sb-cindexboost
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/rbase_%A_%a.out
#SBATCH --exclude=node1,node10,node11,node12,node13,node14,node15,node16,node17,node18,node19,node2,node20,node3,node4,node5,node6,node7,node8,node9
set -euo pipefail
export NEEDS_R=1  # slurm/env.sh loads the R modules instead of Python
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
IFS=$'\t' read -r ds split <<< "$line"
echo "[$(date)] $ds $split"
STABSEL_CORES=${SLURM_CPUS_PER_TASK:-1} Rscript r/cindex_boost_stabsel.R "$ds" "$split"
