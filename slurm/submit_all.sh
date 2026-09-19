#!/bin/bash
# Submit the complete rerun (review points 1-4) as dependent SLURM jobs.
#
#   bash slurm/submit_all.sh              # from the repository root, on the login node
#
#   selection array ─┐
#   [B] (R) array   ─┼─> evaluation array (per dataset) ─┐
#                    └─> stability array  (per dataset) ─┴─> report job
#
# Only selections without a finished record are submitted, so after time-outs or
# failures simply run this script again. Environment variables:
#   PAR=200            max. simultaneously running tasks per array
#   MAX_ARRAY=...      max. array size (default: read from scontrol, else 1001)
#   STAB_B=20          half-samples per method in the stability analysis
#   STAB_METHODS="..." methods for the half-sample analysis
#   SKIP_R=1           do not submit the [B] (R) jobs
#   DATASETS="a b"     restrict everything to these datasets
set -euo pipefail
cd "$(dirname "$0")/.."
source slurm/env.sh
mkdir -p slurm/logs slurm/tasks

# shellcheck disable=SC2086
python slurm/make_tasks.py ${DATASETS:+--datasets $DATASETS}

PAR=${PAR:-200}
if [[ -z "${MAX_ARRAY:-}" ]]; then
  MAX_ARRAY=$(scontrol show config 2>/dev/null | awk -F= '/^MaxArraySize/ {gsub(/ /, "", $2); print $2}')
  MAX_ARRAY=${MAX_ARRAY:-1001}
fi
export STAB_B=${STAB_B:-20}
export STAB_METHODS=${STAB_METHODS:-"SHAPBoost SHAPBoost-noRW GainBoost Lasso Forward-CV XGBoost"}

# submit_array <sbatch file> <task list> [sbatch args...]
# Splits long lists into several arrays of at most MAX_ARRAY-1 tasks (OFFSET tells
# each array where its lines start). Prints ":id1:id2..." (empty if nothing to do).
submit_array() {
  local file=$1 list=$2; shift 2
  local n per off=0 cnt id ids=""
  n=$(wc -l < "$list")
  per=$((MAX_ARRAY - 1))
  while (( off < n )); do
    cnt=$(( n - off < per ? n - off : per ))
    # shellcheck disable=SC2086
    id=$(sbatch --parsable $SBATCH_ACCOUNT_ARGS --array=0-$((cnt - 1))%"$PAR" \
         --export=ALL,TASK_LIST="$list",OFFSET="$off" "$@" "$file")
    ids="$ids:${id%%;*}"
    off=$((off + cnt))
  done
  echo "$ids"
}
dep() { [[ -n "$1" ]] && echo "--dependency=afterany$1" || true; }

SEL=$(submit_array slurm/select.sbatch slurm/tasks/selection.tsv)
RB=""
if [[ -z "${SKIP_R:-}" ]]; then
  RB=$(submit_array slurm/r_baseline.sbatch slurm/tasks/r_baseline.tsv)
fi
# shellcheck disable=SC2046
EVAL=$(submit_array slurm/evaluate.sbatch slurm/tasks/datasets.tsv $(dep "$SEL$RB"))
# shellcheck disable=SC2046
STAB=$(submit_array slurm/stability.sbatch slurm/tasks/datasets.tsv $(dep "$SEL$RB"))
# shellcheck disable=SC2046,SC2086
REP=$(sbatch --parsable $SBATCH_ACCOUNT_ARGS $(dep "$EVAL$STAB") slurm/report.sbatch)

echo "selection: ${SEL:-none}  [B]: ${RB:-none}  evaluation: ${EVAL}  stability: ${STAB}  report: ${REP%%;*}"
echo "logs: slurm/logs/   results: results/   figures: plots/   tables: results/tables/"
