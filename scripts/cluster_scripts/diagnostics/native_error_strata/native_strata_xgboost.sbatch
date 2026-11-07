#!/bin/bash
#SBATCH --job-name=XGBErrStrata
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8g
#SBATCH --time=00:30:00
#SBATCH --constraint="scratch&work"
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# Bins the existing native XGBoost predictions for the CAMEO proteins at all 48 layers,
# tests the stated claims protein by protein, and draws the figures and protein page.
# Compares with the OLS run on the same proteins. No model is run, and no boosters or
# pair tensors are opened. A resubmission skips the scan once its output exists.
module reset
module load python
source ~/protein_venv/bin/activate

export CSV_ROOT=${CSV_ROOT:-/work/hdd/bfna/ikhire}
export NATIVE_ERROR_STRATA_PROBE=shared_xgboost_base256_fixed_s4
export NATIVE_ERROR_STRATA_COHORT=cameo
export NATIVE_ERROR_STRATA_OUTPUT_ROOT=${NATIVE_ERROR_STRATA_OUTPUT_ROOT:-$HOME/diagnostic_experiments/native_error_strata_xgboost_v1}
SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
python3 "$SCRIPT_DIR/canonical_storage.py" --check-write "$NATIVE_ERROR_STRATA_OUTPUT_ROOT" || exit $?
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export MPLCONFIGDIR=$NATIVE_ERROR_STRATA_OUTPUT_ROOT/.matplotlib
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

stamp() { echo "[$(date -u +%FT%TZ)] $*"; }
run() { stamp "stage $1"; srun python3 -u "$SCRIPT_DIR/$1" || { stamp "failed $1"; exit 1; }; }
stamp "start host=$(hostname)"
if [ ! -f "$NATIVE_ERROR_STRATA_OUTPUT_ROOT/data/protein_strata_cameo.npz" ]; then
    run build_protein_strata.py
fi
run build_cameo_strata.py
run prepare_protein_page.py
run make_protein_page.py
run plot_strata.py
run compare_to_ols.py
stamp "end"
