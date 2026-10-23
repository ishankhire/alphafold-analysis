#!/bin/bash
#SBATCH --job-name=CVSharedXGBAgg
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8g
#SBATCH --time=00:30:00
#SBATCH --constraint="scratch&work"
#SBATCH -o aggregate_shared_xgboost_5fold_%j.out
#SBATCH -e aggregate_shared_xgboost_5fold_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ikhire3@gatech.edu

set -euo pipefail

module reset
module load python
source ~/protein_venv/bin/activate
module list

export CSV_ROOT=${CSV_ROOT:-/work/hdd/bfna/ikhire}
SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
python3 "$SCRIPT_DIR/canonical_storage.py" --check-write "$CSV_ROOT/regression_5fold_v1" || exit $?
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=$CSV_ROOT/regression_5fold_v1/shared_xgboost_base256_fixed_s4/logs
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/aggregate_${RUN_TIMESTAMP}_${SLURM_JOB_ID}.log") 2>&1

START_SECONDS=$SECONDS
echo "[$(date --iso-8601=seconds)] fixed S4 shared XGBoost aggregation starting on $(hostname)"

srun python3 -u "$SCRIPT_DIR/aggregate_shared_xgboost_base256_fixed_s4.py"
echo "[$(date --iso-8601=seconds)] fixed S4 shared XGBoost aggregation finished; elapsed_seconds=$((SECONDS - START_SECONDS))"
