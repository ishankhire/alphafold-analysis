#!/bin/bash
#SBATCH --job-name=PredDistPrepare
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2g
#SBATCH --time=00:15:00
#SBATCH --constraint="scratch&work"
#SBATCH -o /work/hdd/bekh/ikhire/regression_5fold_v1/predicted_distances_5fold_v1/logs/prepare-%j.out
#SBATCH -e /work/hdd/bekh/ikhire/regression_5fold_v1/predicted_distances_5fold_v1/logs/prepare-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ikhire3@gatech.edu

set -euo pipefail
module reset
module load python
source ~/protein_venv/bin/activate
module list
export CSV_ROOT=${CSV_ROOT:-$HOME/csv_files}
export PREDICTED_DISTANCE_OUTPUT_ROOT=${PREDICTED_DISTANCE_OUTPUT_ROOT:-/work/hdd/bekh/ikhire/regression_5fold_v1/predicted_distances_5fold_v1}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
srun python3 -u "$SCRIPT_DIR/prepare_predicted_distance_export.py"
