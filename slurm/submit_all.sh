#!/bin/bash
# Submit the complete rerun (review points 1-4) as dependent SLURM jobs.
#
#   bash slurm/submit_all.sh              # on the login node (from any directory)
#   sbatch slurm/submit_all.sh            # also works (runs the submission as a job)
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
# The same options can be given as arguments, which also works on clusters that do
# not pass environment variables into jobs (recommended with sbatch):
#   sbatch slurm/submit_all.sh --datasets "aids whas500" --methods "SHAPBoost" --selection-only
#
#   DATASETS="a b"     restrict everything to these datasets
#   METHODS="a b"      restrict selection to these methods (CIndexBoost-StabSel = [B])
#   SELECTION_ONLY=1   submit only selection and [B]; no evaluation/stability/report
#                      (use for targeted resubmissions while other jobs still run;
#                       afterwards run submit_all.sh once more without it)
#SBATCH --job-name=sb-submit
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=sb-submit_%j.out
set -euo pipefail
# Works with "bash slurm/submit_all.sh" and with "sbatch slurm/submit_all.sh":
# under sbatch this script runs as a copy in SLURM's spool directory, so the
# repository is taken from the path of the submitted script instead of $0.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  script=$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/^.*[[:space:]]Command=//p' | head -n 1)
  [[ "$script" = /* ]] || script="$SLURM_SUBMIT_DIR/$script"
else
  script=$0
fi
cd "$(dirname "$script")/.."
REPO_DIR=$(pwd)
[[ -f slurm/env.sh ]] || { echo "slurm/env.sh not found in $REPO_DIR" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case $1 in
    --datasets)       DATASETS=$2; shift 2 ;;
    --methods)        METHODS=$2; shift 2 ;;
    --selection-only) SELECTION_ONLY=1; shift ;;
    --skip-r)         SKIP_R=1; shift ;;
    *) echo "unknown option: $1 (use --datasets, --methods, --selection-only, --skip-r)" >&2
       exit 1 ;;
  esac
done
echo "settings: DATASETS='${DATASETS:-all}' METHODS='${METHODS:-all}'" \
     "SELECTION_ONLY='${SELECTION_ONLY:-}' SKIP_R='${SKIP_R:-}'"
set +u  # module/conda scripts often use unset variables
source slurm/env.sh
set -u
# Every job gets the repository path and runs (and writes its logs) there,
# independent of the directory this script was started from.
JOB_ARGS="--chdir=$REPO_DIR"
mkdir -p slurm/logs slurm/tasks

# shellcheck disable=SC2086
python slurm/make_tasks.py ${DATASETS:+--datasets $DATASETS} ${METHODS:+--methods $METHODS}

PAR=${PAR:-200}
if [[ -z "${MAX_ARRAY:-}" ]]; then
  # "|| true": scontrol may be missing or restricted; never abort silently here
  MAX_ARRAY=$( (scontrol show config 2>/dev/null || true) |
               awk -F= '/^MaxArraySize/ {gsub(/ /, "", $2); print $2}')
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
  mkdir -p slurm/tasks/submitted
  while (( off < n )); do
    cnt=$(( n - off < per ? n - off : per ))
    # Each array gets its OWN copy of its lines: the jobs read this file when they
    # start, and slurm/tasks/*.tsv is rewritten by every later submit_all.sh run.
    chunk="slurm/tasks/submitted/$(date +%Y%m%d-%H%M%S)_$(basename "$list" .tsv)_$off.tsv"
    sed -n "$((off + 1)),$((off + cnt))p" "$list" > "$chunk"
    # shellcheck disable=SC2086
    id=$(sbatch --parsable $SBATCH_ACCOUNT_ARGS $JOB_ARGS --array=0-$((cnt - 1))%"$PAR" \
         --export=ALL,REPO_DIR="$REPO_DIR",TASK_LIST="$chunk",OFFSET=0 "$@" "$file")
    id=${id%%;*}
    ln -f "$chunk" "slurm/tasks/submitted/$id.tsv"  # for check_status.py
    ids="$ids:$id"
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
if [[ -n "${SELECTION_ONLY:-}" ]]; then
  echo "selection: ${SEL:-none}  [B]: ${RB:-none}  (SELECTION_ONLY: no evaluation/stability/report)"
  exit 0
fi
# shellcheck disable=SC2046
EVAL=$(submit_array slurm/evaluate.sbatch slurm/tasks/datasets.tsv $(dep "$SEL$RB"))
# shellcheck disable=SC2046
STAB=$(submit_array slurm/stability.sbatch slurm/tasks/datasets.tsv $(dep "$SEL$RB"))
# shellcheck disable=SC2046,SC2086
REP=$(sbatch --parsable $SBATCH_ACCOUNT_ARGS $JOB_ARGS --export=ALL,REPO_DIR="$REPO_DIR" \
      $(dep "$EVAL$STAB") slurm/report.sbatch)

echo "selection: ${SEL:-none}  [B]: ${RB:-none}  evaluation: ${EVAL}  stability: ${STAB}  report: ${REP%%;*}"
echo "logs: slurm/logs/   results: results/   figures: plots/   tables: results/tables/"
