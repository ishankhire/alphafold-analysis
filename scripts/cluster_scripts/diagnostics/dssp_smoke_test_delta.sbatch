#!/bin/bash
#SBATCH --job-name=DSSPSmoke
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4g
#SBATCH --time=00:30:00
#SBATCH --constraint="scratch"
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ikhire3@gatech.edu

module reset
module load python
source ~/protein_venv/bin/activate

# Change this ID for another final predicted PDB. Numeric IDs resolve via the
# ECOD root; all other IDs resolve via the CAMEO root.
export DSSP_SAMPLE_PROTEIN=6tf4_A
export DSSP_EXECUTABLE=$HOME/dssp_env/bin/mkdssp
export DSSP_SMOKE_OUTPUT_DIR=$HOME/csv_files/dssp_smoke_test
export CAMEO_PDB_ROOT=/work/hdd/bekh/thayes/openfold_reps_outputs_sm
export ECOD_PDB_ROOT=/work/hdd/bfna/thayes/ecod/experiments/openfold_reps_outputs_sm

SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
test -x "$DSSP_EXECUTABLE"
"$DSSP_EXECUTABLE" --version
mkdir -p "$DSSP_SMOKE_OUTPUT_DIR"
srun python3 -u "$SCRIPT_DIR/dssp_smoke_test.py"
