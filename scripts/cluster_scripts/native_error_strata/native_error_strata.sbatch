#!/bin/bash
#SBATCH --job-name=NativeErrStrata
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8g
#SBATCH --time=01:00:00
#SBATCH --constraint="scratch&work"
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# Bins existing native OOF predictions by sequence separation and final structure
# distance for all 48 layers, then plots. No model is run. A resubmission skips the
# scan once the checked CSV exists, so it only verifies the targets and redraws.
# Needs CROSS_LAYER_ECOD_EXPERIMENTS_ROOT and the ECOD PDBs, which are currently unavailable.
module reset
module load python
source ~/protein_venv/bin/activate

export CSV_ROOT=${CSV_ROOT:-/work/hdd/bfna/ikhire}
export NATIVE_ERROR_STRATA_OUTPUT_ROOT=${NATIVE_ERROR_STRATA_OUTPUT_ROOT:-$HOME/diagnostic_experiments/native_error_strata_v1}
SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
python3 "$SCRIPT_DIR/canonical_storage.py" --check-write "$NATIVE_ERROR_STRATA_OUTPUT_ROOT" || exit $?
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export MPLCONFIGDIR=$NATIVE_ERROR_STRATA_OUTPUT_ROOT/.matplotlib
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

stamp() { echo "[$(date -u +%FT%TZ)] $*"; }
stamp "start host=$(hostname)"
if [ ! -f "$NATIVE_ERROR_STRATA_OUTPUT_ROOT/data/native_error_strata.csv" ]; then
    srun python3 -u "$SCRIPT_DIR/build_strata.py" || { stamp "build failed"; exit 1; }
fi
srun python3 -u "$SCRIPT_DIR/verify_training_target.py"
check_status=$?
stamp "training target check status=$check_status"
srun python3 -u "$SCRIPT_DIR/plot_strata.py" && srun python3 -u "$SCRIPT_DIR/make_page.py"
plot_status=$?
stamp "end check_status=$check_status plot_status=$plot_status"
if [ "$check_status" -ne 0 ] || [ "$plot_status" -ne 0 ]; then
    exit 1
fi
