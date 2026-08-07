#!/bin/bash
#SBATCH --job-name=Cameo50Interactions
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=04:00:00
#SBATCH --constraint="scratch"
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ikhire3@gatech.edu

module reset
module load python
source ~/protein_venv/bin/activate
export DSSP_EXECUTABLE=$HOME/dssp_env/bin/mkdssp
export PAIR_BLOCKS_ROOT=/work/hdd/bekh/thayes/openfold_reps
export PDB_ROOT=/work/hdd/bekh/thayes/openfold_reps_outputs_sm
export CSV_ROOT=$HOME/csv_files/pair_interaction_classification/cameo50_v1
SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
python3 - <<'PY'
import Bio, numpy, scipy, os
assert os.path.isfile(os.environ['DSSP_EXECUTABLE']), 'missing DSSP executable'
print('preflight:', Bio.__version__, numpy.__version__, scipy.__version__)
PY
test -d "$PAIR_BLOCKS_ROOT" && test -d "$PDB_ROOT"
srun python3 -u "$SCRIPT_DIR/diagnostic_pair_interaction_annotation.py"
