#!/bin/bash
#SBATCH --job-name=SmacofDecompAgg
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8g
#SBATCH --time=01:00:00
#SBATCH --constraint="scratch&work"
#SBATCH -o /work/hdd/bfna/ikhire/diagnostic_experiments/smacof_error_decomposition_5fold_v1/logs/aggregate_smacof_error_decomposition_%j.out
#SBATCH -e /work/hdd/bfna/ikhire/diagnostic_experiments/smacof_error_decomposition_5fold_v1/logs/aggregate_smacof_error_decomposition_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# Collects the layer shards into tables/ after the array (submit with
# --dependency=afterok:<array job>). Refuses to run unless every layer validated.
module reset
module load python
source ~/protein_venv/bin/activate

export CSV_ROOT=${CSV_ROOT:-/work/hdd/bfna/ikhire}
export SMACOF_DECOMPOSITION_OUTPUT_ROOT=${SMACOF_DECOMPOSITION_OUTPUT_ROOT:-$CSV_ROOT/diagnostic_experiments/smacof_error_decomposition_5fold_v1}
SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
python3 "$SCRIPT_DIR/canonical_storage.py" --check-write "$SMACOF_DECOMPOSITION_OUTPUT_ROOT" || exit $?
echo "[$(date -u +%FT%TZ)] start aggregation output=$SMACOF_DECOMPOSITION_OUTPUT_ROOT"
srun python3 -u "$SCRIPT_DIR/aggregate_smacof_error_decomposition.py"
status=$?
echo "[$(date -u +%FT%TZ)] end aggregation status=$status"
exit $status
