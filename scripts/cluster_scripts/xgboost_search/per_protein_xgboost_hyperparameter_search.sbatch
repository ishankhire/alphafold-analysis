#!/bin/bash
#SBATCH --job-name=XGBPerProtSearch
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --array=0-2%3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=10:00:00
#SBATCH --constraint="scratch"
#SBATCH -o per_protein_xgb_search_%A_%a.out
#SBATCH -e per_protein_xgb_search_%A_%a.err
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
LOG_DIR=$CSV_ROOT/xgboost_hyperparameter_search/logs
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/per_protein_search_${RUN_TIMESTAMP}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log") 2>&1

echo "[$(date --iso-8601=seconds)] job starting on $(hostname)"
SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
srun python3 -u "$SCRIPT_DIR/per_protein_xgboost_hyperparameter_search.py"
