#!/bin/bash
#SBATCH --job-name=TMscoreRescore
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2g
#SBATCH --time=02:00:00
#SBATCH --constraint="scratch&work"
#SBATCH --array=0-47%4
#SBATCH -o /work/hdd/bfna/ikhire/regression_5fold_v1/tm_score_optimized_v1/logs/layer-%A_%a.out
#SBATCH -e /work/hdd/bfna/ikhire/regression_5fold_v1/tm_score_optimized_v1/logs/layer-%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ikhire3@gatech.edu

set -euo pipefail
module reset
module load python
source ~/protein_venv/bin/activate
export CSV_ROOT=${CSV_ROOT:-$HOME/csv_files}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
if [[ ${TMSCORE_FINALIZE:-0} == 1 ]]; then
    srun python3 -u "$SCRIPT_DIR/recompute_tm_scores.py" --finalize
else
    srun python3 -u "$SCRIPT_DIR/recompute_tm_scores.py" --layer "$SLURM_ARRAY_TASK_ID"
fi
