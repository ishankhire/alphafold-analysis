#!/bin/bash
#SBATCH --job-name=TriStrata
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2g
#SBATCH --time=00:15:00
#SBATCH --constraint="scratch&work"
#SBATCH --array=0-47%16
#SBATCH -o slurm-%A_%a.out
#SBATCH -e slurm-%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL

# Task i scores the error models on layer i for OLS and XGBoost. No model is run. Every
# task must reproduce the stored strata and violation fractions before writing. Pass the
# notice recipient at submission; see the README.
module reset
module load python
source ~/protein_venv/bin/activate

export CSV_ROOT=${CSV_ROOT:-/work/hdd/bfna/ikhire}
export TRIANGLE_STRATA_OUTPUT_ROOT=${TRIANGLE_STRATA_OUTPUT_ROOT:-$HOME/diagnostic_experiments/triangle_strata_cameo_v1}
SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
python3 "$SCRIPT_DIR/canonical_storage.py" --check-write "$TRIANGLE_STRATA_OUTPUT_ROOT" || exit $?
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "[$(date -u +%FT%TZ)] start task=${SLURM_ARRAY_TASK_ID} output=$TRIANGLE_STRATA_OUTPUT_ROOT"
srun python3 -u "$SCRIPT_DIR/triangle_strata_replay.py"
status=$?
echo "[$(date -u +%FT%TZ)] end task=${SLURM_ARRAY_TASK_ID} status=$status"
exit $status
