#!/bin/bash
#SBATCH --job-name=XLayerXferAgg
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6g
#SBATCH --time=00:30:00
#SBATCH --constraint="scratch&work"
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ikhire3@gatech.edu

module reset
module load python
source ~/protein_venv/bin/activate

export CSV_ROOT=${CSV_ROOT:-/work/hdd/bfna/ikhire}
export CROSS_LAYER_TRANSFER_OUTPUT_ROOT=${CROSS_LAYER_TRANSFER_OUTPUT_ROOT:-$CSV_ROOT/diagnostic_experiments/cross_layer_probe_transfer_v1}
SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
python3 "$SCRIPT_DIR/canonical_storage.py" --check-write "$CROSS_LAYER_TRANSFER_OUTPUT_ROOT" || exit $?
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "[$(date -u +%FT%TZ)] start aggregation output=$CROSS_LAYER_TRANSFER_OUTPUT_ROOT"
srun python3 -u "$SCRIPT_DIR/aggregate_cross_layer_transfer.py"
status=$?
echo "[$(date -u +%FT%TZ)] end aggregation status=$status"
exit $status
