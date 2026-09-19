#!/bin/bash
# C-index boosting + stability selection [B] for one (dataset, split) line of
# $TASK_LIST. The 100 stability-selection refits run on the job's cores.
#SBATCH --job-name=sb-cindexboost
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/rbase_%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh

line=$(sed -n "$(( ${OFFSET:-0} + SLURM_ARRAY_TASK_ID + 1 ))p" "$TASK_LIST")
IFS=$'\t' read -r ds split <<< "$line"
echo "[$(date)] $ds $split"
STABSEL_CORES=${SLURM_CPUS_PER_TASK:-1} Rscript r/cindex_boost_stabsel.R "$ds" "$split"
