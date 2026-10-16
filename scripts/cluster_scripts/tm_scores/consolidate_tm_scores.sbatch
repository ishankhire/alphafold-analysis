#!/bin/bash
#SBATCH --job-name=TMscoreConsolidate
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3g
#SBATCH --time=00:10:00
#SBATCH --constraint="scratch&work"
#SBATCH -o /work/hdd/bfna/ikhire/regression_5fold_v1/tm_score_optimized_v1/logs/consolidate-%j.out
#SBATCH -e /work/hdd/bfna/ikhire/regression_5fold_v1/tm_score_optimized_v1/logs/consolidate-%j.err
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
srun python3 -u "$SCRIPT_DIR/consolidate_tm_scores.py"
