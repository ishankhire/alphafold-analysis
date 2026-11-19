#!/bin/bash
#SBATCH --job-name=ProbeLayerSim
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=01:00:00
#SBATCH --constraint="scratch&work"
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# Weight, prediction and error similarity between layers for the frozen probes.
# Reads stored coefficients, out-of-fold predictions and final structure PDBs; fits nothing.
# Pass the notice recipient at submission; see the README.
module reset
module load python
source ~/protein_venv/bin/activate

export CSV_ROOT=${CSV_ROOT:-/work/hdd/bfna/ikhire}
export PROBE_LAYER_SIMILARITY_OUTPUT_ROOT=${PROBE_LAYER_SIMILARITY_OUTPUT_ROOT:-$HOME/diagnostic_experiments/probe_layer_similarity_v1}
SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
python3 "$SCRIPT_DIR/canonical_storage.py" --check-write "$PROBE_LAYER_SIMILARITY_OUTPUT_ROOT" || exit $?
# One process per CPU, each single-threaded.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

echo "[$(date -u +%FT%TZ)] start output=$PROBE_LAYER_SIMILARITY_OUTPUT_ROOT"
srun python3 -u "$SCRIPT_DIR/probe_layer_similarity.py"
status=$?
echo "[$(date -u +%FT%TZ)] end status=$status"
exit $status
