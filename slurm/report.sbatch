#!/bin/bash
# All figures (plots/) and tables (results/tables/) for the paper and the response.
#SBATCH --job-name=sb-report
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/report_%j.out
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
python evaluation_experiment1.py
python visualize_results.py
python plot_operating_points.py
python report_ablation.py
