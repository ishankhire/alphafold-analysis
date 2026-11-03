#!/bin/bash
#SBATCH --job-name=XLayerXfer
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12g
#SBATCH --time=02:00:00
#SBATCH --constraint="scratch&work"
#SBATCH --array=0-9%10
#SBATCH -o slurm-%A_%a.out
#SBATCH -e slurm-%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ikhire3@gatech.edu

# Task i replays all ten frozen shared-OLS source probes on target layer
# TRANSFER_LAYERS[i] = (22,23,24,25,26,43,44,45,46,47)[i].
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

echo "[$(date -u +%FT%TZ)] start task=${SLURM_ARRAY_TASK_ID} output=$CROSS_LAYER_TRANSFER_OUTPUT_ROOT"
srun python3 -u "$SCRIPT_DIR/cross_layer_transfer_replay.py"
status=$?
echo "[$(date -u +%FT%TZ)] end task=${SLURM_ARRAY_TASK_ID} status=$status"
exit $status
