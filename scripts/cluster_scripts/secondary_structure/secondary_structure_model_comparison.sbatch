#!/bin/bash
#SBATCH --job-name=secondary_structure_comparison
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=secondary_structure_%x_%A_%a.out
#SBATCH --error=secondary_structure_%x_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ikhire3@gatech.edu
#SBATCH --constraint=work

set -euo pipefail

SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
module reset
module load python
source "$HOME/protein_venv/bin/activate"
export CSV_ROOT=${CSV_ROOT:-$HOME/csv_files}
export SHARED_GLOBAL_OLS_ROOT=${SHARED_GLOBAL_OLS_ROOT:-$CSV_ROOT/general_base256_combined}
export SEQUENCE_GATED_OLS_ROOT=${SEQUENCE_GATED_OLS_ROOT:-$CSV_ROOT/gated_linear_probes/consolidated/analysis_1_sequence_gated}
export PER_PROTEIN_OLS_ROOT=${PER_PROTEIN_OLS_ROOT:-$CSV_ROOT/per_protein_base256_combined}
export SHARED_XGBOOST_ROOT=${SHARED_XGBOOST_ROOT:-$CSV_ROOT/shared_xgboost_base256_combined}
export SECONDARY_STRUCTURE_OUTPUT_DIR=${SECONDARY_STRUCTURE_OUTPUT_DIR:-$CSV_ROOT/secondary_structure_model_comparison}
export CAMEO_PAIR_BLOCKS_ROOT=${CAMEO_PAIR_BLOCKS_ROOT:-/work/hdd/bekh/thayes/openfold_reps}
export CAMEO_PDB_ROOT=${CAMEO_PDB_ROOT:-/work/hdd/bekh/thayes/openfold_reps_outputs_sm}
export ECOD_PAIR_BLOCKS_ROOT=${ECOD_PAIR_BLOCKS_ROOT:-/work/hdd/bfna/thayes/ecod/experiments/representation_saves}
export ECOD_PDB_ROOT=${ECOD_PDB_ROOT:-/work/hdd/bfna/thayes/ecod/experiments/openfold_reps_outputs_sm}
export DSSP_EXECUTABLE=${DSSP_EXECUTABLE:-$HOME/dssp_env/bin/mkdssp}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export MPLCONFIGDIR=${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-local}}
mkdir -p "$MPLCONFIGDIR"
test -x "$DSSP_EXECUTABLE"

echo "[$(date --iso-8601=seconds)] stage=$SECONDARY_STRUCTURE_STAGE task=${SLURM_ARRAY_TASK_ID:-none}"
srun python3 -u "$SCRIPT_DIR/secondary_structure_model_comparison.py"
echo "[$(date --iso-8601=seconds)] completed"
