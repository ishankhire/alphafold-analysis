#!/bin/bash
#SBATCH --job-name=XLayerAllAgg
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16g
#SBATCH --time=01:00:00
#SBATCH --constraint="scratch&work"
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# Pass the notice recipient at submission; see the README.
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

echo "[$(date -u +%FT%TZ)] start aggregation output=$ALL_LAYER_TRANSFER_OUTPUT_ROOT"
srun python3 -u "$SCRIPT_DIR/aggregate_all_layer_transfer.py"
status=$?
echo "[$(date -u +%FT%TZ)] end aggregation status=$status"
exit $status
