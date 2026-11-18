#!/bin/bash
#SBATCH --job-name=ProbeLayerSimFig
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8g
#SBATCH --time=00:30:00
#SBATCH --constraint="scratch&work"
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# Draws the report figures from the similarity tables and the all-layer transfer matrix.
# Pass the notice recipient at submission; see the README.
module reset
module load python
source ~/protein_venv/bin/activate

export PROBE_LAYER_SIMILARITY_OUTPUT_ROOT=${PROBE_LAYER_SIMILARITY_OUTPUT_ROOT:-$HOME/diagnostic_experiments/probe_layer_similarity_v1}
export ALL_LAYER_TRANSFER_OUTPUT_ROOT=${ALL_LAYER_TRANSFER_OUTPUT_ROOT:-$HOME/diagnostic_experiments/cross_layer_probe_transfer_all_layers_v1}
export MPLCONFIGDIR=$PROBE_LAYER_SIMILARITY_OUTPUT_ROOT/.matplotlib
SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

echo "[$(date -u +%FT%TZ)] start figures output=$PROBE_LAYER_SIMILARITY_OUTPUT_ROOT"
srun python3 -u "$SCRIPT_DIR/make_probe_layer_similarity_figures.py"
status=$?
echo "[$(date -u +%FT%TZ)] end figures status=$status"
exit $status
