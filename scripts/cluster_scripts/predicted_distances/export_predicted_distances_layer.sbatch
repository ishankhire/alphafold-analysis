#!/bin/bash
#SBATCH --job-name=PredDistLayer
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12g
#SBATCH --time=02:00:00
#SBATCH --constraint="scratch&work"
#SBATCH --array=0-47%4
#SBATCH -o /work/hdd/bfna/ikhire/regression_5fold_v1/predicted_distances_5fold_v1/logs/layer-%A_%a.out
#SBATCH -e /work/hdd/bfna/ikhire/regression_5fold_v1/predicted_distances_5fold_v1/logs/layer-%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ikhire3@gatech.edu

set -euo pipefail
module reset
module load python
source ~/protein_venv/bin/activate
module list
export CSV_ROOT=${CSV_ROOT:-$HOME/csv_files}
export PREDICTED_DISTANCE_OUTPUT_ROOT=${PREDICTED_DISTANCE_OUTPUT_ROOT:-/work/hdd/bfna/ikhire/regression_5fold_v1/predicted_distances_5fold_v1}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
srun python3 -u "$SCRIPT_DIR/export_predicted_distances_layer.py"
