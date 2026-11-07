#!/bin/bash
#SBATCH --job-name=GainPosition
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12g
#SBATCH --time=00:15:00
#SBATCH --constraint="scratch&work"
#SBATCH --array=0-47%16
#SBATCH -o slurm-%A_%a.out
#SBATCH -e slurm-%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL

# Task i replays all source probes on target layer i for the CAMEO proteins and must
# reproduce the all-layer transfer shard before writing. Pass the notice recipient at
# submission; see the README.
module reset
module load python
source ~/protein_venv/bin/activate

export CSV_ROOT=${CSV_ROOT:-/work/hdd/bfna/ikhire}
export GAIN_POSITION_OUTPUT_ROOT=${GAIN_POSITION_OUTPUT_ROOT:-$HOME/diagnostic_experiments/gain_and_chain_position_cameo_v1}
SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
python3 "$SCRIPT_DIR/canonical_storage.py" --check-write "$GAIN_POSITION_OUTPUT_ROOT" || exit $?
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "[$(date -u +%FT%TZ)] start task=${SLURM_ARRAY_TASK_ID} output=$GAIN_POSITION_OUTPUT_ROOT"
srun python3 -u "$SCRIPT_DIR/gain_position_replay.py"
status=$?
echo "[$(date -u +%FT%TZ)] end task=${SLURM_ARRAY_TASK_ID} status=$status"
exit $status
