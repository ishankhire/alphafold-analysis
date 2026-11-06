#!/bin/bash
#SBATCH --job-name=XLayerAll
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12g
#SBATCH --time=01:00:00
#SBATCH --constraint="scratch&work"
#SBATCH --array=0-47%16
#SBATCH -o slurm-%A_%a.out
#SBATCH -e slurm-%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL

# Task i replays all 48 frozen shared-OLS source probes on target layer i.
# ALL_LAYER_COHORT must name the cohort, cameo or full; full also needs
# CROSS_LAYER_ECOD_EXPERIMENTS_ROOT. Pass the notice recipient at submission; see the README.
module reset
module load python
source ~/protein_venv/bin/activate

export CSV_ROOT=${CSV_ROOT:-/work/hdd/bfna/ikhire}
export ALL_LAYER_COHORT=${ALL_LAYER_COHORT:?Set ALL_LAYER_COHORT to cameo or full}
if [ "$ALL_LAYER_COHORT" = full ]; then NAME=cross_layer_probe_transfer_all_layers_v1
else NAME=cross_layer_probe_transfer_all_layers_${ALL_LAYER_COHORT}_v1; fi
export ALL_LAYER_TRANSFER_OUTPUT_ROOT=${ALL_LAYER_TRANSFER_OUTPUT_ROOT:-$HOME/diagnostic_experiments/$NAME}
SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
python3 "$SCRIPT_DIR/canonical_storage.py" --check-write "$ALL_LAYER_TRANSFER_OUTPUT_ROOT" || exit $?
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "[$(date -u +%FT%TZ)] start task=${SLURM_ARRAY_TASK_ID} output=$ALL_LAYER_TRANSFER_OUTPUT_ROOT"
srun python3 -u "$SCRIPT_DIR/all_layer_transfer_replay.py"
status=$?
echo "[$(date -u +%FT%TZ)] end task=${SLURM_ARRAY_TASK_ID} status=$status"
exit $status
