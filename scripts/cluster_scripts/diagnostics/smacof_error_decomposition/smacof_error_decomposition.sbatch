#!/bin/bash
#SBATCH --job-name=SmacofDecomp
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32g
#SBATCH --time=06:00:00
#SBATCH --constraint="scratch&work"
#SBATCH --array=0-47%16
#SBATCH -o /work/hdd/bfna/ikhire/diagnostic_experiments/smacof_error_decomposition_5fold_v1/logs/smacof_error_decomposition_%A_%a.out
#SBATCH -e /work/hdd/bfna/ikhire/diagnostic_experiments/smacof_error_decomposition_5fold_v1/logs/smacof_error_decomposition_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL

# Task i fits layer i: observed decoded distances and the triangle experiment's B10 controls
# (4 probes, 552 proteins, 28 matrices per protein), parallel over proteins. Nothing is written
# unless every gate passes. Pass the notice recipient at submission; see
# smacof_error_decomposition_README.md.
module reset
module load python
source ~/protein_venv/bin/activate

export CSV_ROOT=${CSV_ROOT:-/work/hdd/bfna/ikhire}
export SMACOF_DECOMPOSITION_OUTPUT_ROOT=${SMACOF_DECOMPOSITION_OUTPUT_ROOT:-$CSV_ROOT/diagnostic_experiments/smacof_error_decomposition_5fold_v1}
SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
python3 "$SCRIPT_DIR/canonical_storage.py" --check-write "$SMACOF_DECOMPOSITION_OUTPUT_ROOT" || exit $?
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=${TMPDIR:-/tmp}

echo "[$(date -u +%FT%TZ)] start task=${SLURM_ARRAY_TASK_ID} output=$SMACOF_DECOMPOSITION_OUTPUT_ROOT rel_tol=${SMACOF_REL_TOL:-default} max_iter=${SMACOF_MAX_ITER:-default}"
srun python3 -u "$SCRIPT_DIR/smacof_error_decomposition.py"
status=$?
echo "[$(date -u +%FT%TZ)] end task=${SLURM_ARRAY_TASK_ID} status=$status"
exit $status
