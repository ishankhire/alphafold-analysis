#!/bin/bash
#SBATCH --job-name=XGBFinalAggregate
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8g
#SBATCH --time=01:00:00
#SBATCH --constraint="scratch"
#SBATCH -o shared_xgb_aggregate_%j.out
#SBATCH -e shared_xgb_aggregate_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ikhire3@gatech.edu

module reset
module load python
source ~/protein_venv/bin/activate
module list

export CSV_ROOT=$HOME/csv_files
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=$CSV_ROOT/shared_xgboost_base256_combined_work/logs
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/aggregate_${RUN_TIMESTAMP}_${SLURM_JOB_ID}.log") 2>&1

echo "[$(date --iso-8601=seconds)] aggregation starting on $(hostname)"
SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
srun python3 -u "$SCRIPT_DIR/aggregate_shared_xgboost_final.py"
