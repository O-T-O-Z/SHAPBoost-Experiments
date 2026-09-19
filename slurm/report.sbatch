#!/bin/bash
# All figures (plots/) and tables (results/tables/) for the paper and the response.
#SBATCH --job-name=sb-report
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/report_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
python evaluation_experiment1.py
python visualize_results.py
python plot_operating_points.py
python report_ablation.py
