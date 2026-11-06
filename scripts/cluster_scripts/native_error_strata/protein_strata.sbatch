#!/bin/bash
#SBATCH --job-name=ProteinErrStrata
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

# Bins existing native OOF predictions per protein for all 48 layers, tests the stated
# claims protein by protein, and builds the web page. No model is run. A
# resubmission skips the scan once protein_strata_cameo.npz exists.
module reset
module load python
source ~/protein_venv/bin/activate

export CSV_ROOT=${CSV_ROOT:-/work/hdd/bfna/ikhire}
export NATIVE_ERROR_STRATA_OUTPUT_ROOT=${NATIVE_ERROR_STRATA_OUTPUT_ROOT:-$HOME/diagnostic_experiments/native_error_strata_v1}
SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
python3 "$SCRIPT_DIR/canonical_storage.py" --check-write "$NATIVE_ERROR_STRATA_OUTPUT_ROOT" || exit $?
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

stamp() { echo "[$(date -u +%FT%TZ)] $*"; }
stamp "start host=$(hostname)"
if [ ! -f "$NATIVE_ERROR_STRATA_OUTPUT_ROOT/data/protein_strata_cameo.npz" ]; then
    srun python3 -u "$SCRIPT_DIR/build_protein_strata.py" || { stamp "build failed"; exit 1; }
fi
srun python3 -u "$SCRIPT_DIR/prepare_protein_page.py" || { stamp "prepare failed"; exit 1; }
srun python3 -u "$SCRIPT_DIR/make_protein_page.py"
status=$?
stamp "end page_status=$status"
exit $status
