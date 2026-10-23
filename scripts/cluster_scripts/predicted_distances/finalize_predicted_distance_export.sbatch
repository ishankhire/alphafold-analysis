#!/bin/bash
#SBATCH --job-name=PredDistFinalize
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3g
#SBATCH --time=04:30:00
#SBATCH --constraint="scratch&work"
#SBATCH -o predicted_distances_finalize-%j.out
#SBATCH -e predicted_distances_finalize-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ikhire3@gatech.edu

set -euo pipefail
module reset
module load python
source ~/protein_venv/bin/activate
module list
export CSV_ROOT=${CSV_ROOT:-/work/hdd/bfna/ikhire}
SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
python3 "$SCRIPT_DIR/canonical_storage.py" --check-write "${PREDICTED_DISTANCE_OUTPUT_ROOT:-$CSV_ROOT/regression_5fold_v1/predicted_distances_5fold_v1}" || exit $?
export PREDICTED_DISTANCE_OUTPUT_ROOT=${PREDICTED_DISTANCE_OUTPUT_ROOT:-/work/hdd/bfna/ikhire/regression_5fold_v1/predicted_distances_5fold_v1}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun python3 -u "$SCRIPT_DIR/finalize_predicted_distance_export.py"
