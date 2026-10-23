#!/bin/bash
#SBATCH --job-name=CVSeqGateOLS
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --array=0-239%8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120g
#SBATCH --time=03:00:00
#SBATCH --constraint="scratch&work"
#SBATCH -o sequence_gated_ols_5fold_%A_%a.out
#SBATCH -e sequence_gated_ols_5fold_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ikhire3@gatech.edu

module reset
module load python
source ~/protein_venv/bin/activate
module list

export CSV_ROOT=${CSV_ROOT:-/work/hdd/bfna/ikhire}
SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
python3 "$SCRIPT_DIR/canonical_storage.py" --check-write "$CSV_ROOT/regression_5fold_v1" || exit $?
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=$CSV_ROOT/regression_5fold_v1/sequence_gated_ols_base256/logs
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/task_${RUN_TIMESTAMP}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log") 2>&1

echo "[$(date --iso-8601=seconds)] sequence-gated OLS fold/layer task starting on $(hostname)"

srun python3 -u "$SCRIPT_DIR/sequence_gated_ols_base256_fold_layer.py"
