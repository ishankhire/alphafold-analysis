#!/bin/bash
#SBATCH --job-name=CVPerProtOLSAgg
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8g
#SBATCH --time=01:00:00
#SBATCH --constraint="scratch"
#SBATCH -o aggregate_per_protein_ols_5fold_%j.out
#SBATCH -e aggregate_per_protein_ols_5fold_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ikhire3@gatech.edu

set -euo pipefail

module reset
module load python
source ~/protein_venv/bin/activate
module list

export CSV_ROOT=${CSV_ROOT:-$HOME/csv_files}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=$CSV_ROOT/regression_5fold_v1/per_protein_ols_base256/logs
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/aggregate_${RUN_TIMESTAMP}_${SLURM_JOB_ID}.log") 2>&1

START_SECONDS=$SECONDS
echo "[$(date --iso-8601=seconds)] per-protein OLS aggregation starting on $(hostname)"
SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
srun python3 -u "$SCRIPT_DIR/aggregate_per_protein_ols_base256.py"
echo "[$(date --iso-8601=seconds)] per-protein OLS aggregation finished; elapsed_seconds=$((SECONDS - START_SECONDS))"
