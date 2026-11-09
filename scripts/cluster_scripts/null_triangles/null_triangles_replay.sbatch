#!/bin/bash
#SBATCH --job-name=NullTri
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16g
#SBATCH --time=01:00:00
#SBATCH --constraint="scratch&work"
#SBATCH --array=0-47%16
#SBATCH -o /work/hdd/bfna/ikhire/main_experiments/null_triangles_5fold_v1/logs/null_triangles_replay_%A_%a.out
#SBATCH -e /work/hdd/bfna/ikhire/main_experiments/null_triangles_5fold_v1/logs/null_triangles_replay_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL

# Task i scores observed predictions and Nulls A/B on every triangle of layer i (4 probes,
# 552 proteins: 154 CAMEO + 398 ECOD), parallel over proteins. Nothing is written unless
# the parity and histogram gates pass. Pass the notice recipient at submission; see
# null_triangles_README.md.
module reset
module load python
source ~/protein_venv/bin/activate

export CSV_ROOT=${CSV_ROOT:-/work/hdd/bfna/ikhire}
export NULL_TRIANGLES_OUTPUT_ROOT=${NULL_TRIANGLES_OUTPUT_ROOT:-$CSV_ROOT/main_experiments/null_triangles_5fold_v1}
SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
python3 "$SCRIPT_DIR/canonical_storage.py" --check-write "$NULL_TRIANGLES_OUTPUT_ROOT" || exit $?
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=${TMPDIR:-/tmp}

echo "[$(date -u +%FT%TZ)] start task=${SLURM_ARRAY_TASK_ID} output=$NULL_TRIANGLES_OUTPUT_ROOT"
srun python3 -u "$SCRIPT_DIR/null_triangles_replay.py"
status=$?
echo "[$(date -u +%FT%TZ)] end task=${SLURM_ARRAY_TASK_ID} status=$status"
exit $status
