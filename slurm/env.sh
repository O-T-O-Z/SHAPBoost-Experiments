# Sourced by every SLURM job and by submit_all.sh. Edit for your cluster.
#
# Python environment with requirements.txt installed, e.g.:
#   module load Python/3.11
#   source "$HOME/venvs/shapboost/bin/activate"
# R with mboost, stabs, jsonlite and survival (only needed for the [B] jobs), e.g.:
#   module load R/4.3
#
# Optional: account / partition for all jobs (leave empty to use cluster defaults)
export SBATCH_ACCOUNT_ARGS=""

# Numerical libraries: never use more threads than SLURM gave the job.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

if [[ "${NEEDS_R:-}" == 1 ]]; then
  module load R/4.5.1-gfbf-2025a
  export R_LIBS_USER=$HOME/R/library
else
  module load Python/3.12.3-GCCcore-13.3.0
  source /scratch/p310333/shapboost_venv/bin/activate
fi