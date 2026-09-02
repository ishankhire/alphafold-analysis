#!/bin/bash
#SBATCH --job-name=PairTriAgg
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8g
#SBATCH --time=00:30:00
#SBATCH --output=pair_triangle_aggregate_%j.out
#SBATCH --error=pair_triangle_aggregate_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ikhire3@gatech.edu

set -euo pipefail

SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
module reset
module load python
source "$HOME/protein_venv/bin/activate"
export CSV_ROOT=${CSV_ROOT:-$HOME/csv_files}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "[$(date --iso-8601=seconds)] pair-triangle aggregation starting"
srun python3 -u "$SCRIPT_DIR/aggregate_pair_triangle_dominance.py"
echo "[$(date --iso-8601=seconds)] pair-triangle aggregation completed"
