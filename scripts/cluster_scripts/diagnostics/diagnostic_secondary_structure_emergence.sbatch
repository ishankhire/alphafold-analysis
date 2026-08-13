#!/bin/bash
#SBATCH --job-name=SecStructEmergence
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24g
#SBATCH --time=24:00:00
#SBATCH --array=0-5%3
#SBATCH --constraint="work"
#SBATCH -o slurm-%A_%a.out
#SBATCH -e slurm-%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ikhire3@gatech.edu

module reset
module load python
source ~/protein_venv/bin/activate

module list
echo "job is starting on $(hostname)"
echo "stage=${SUBSTRUCTURE_STAGE:-layer}; array_task=${SLURM_ARRAY_TASK_ID:-unset}"

export INPUT_RESULTS_DIR=${INPUT_RESULTS_DIR:-$HOME/csv_files/general_base256_combined}
export CSV_ROOT=${CSV_ROOT:-$HOME/csv_files}
export CAMEO_PAIR_BLOCKS_ROOT=${CAMEO_PAIR_BLOCKS_ROOT:-/work/hdd/bekh/thayes/openfold_reps}
export CAMEO_PDB_ROOT=${CAMEO_PDB_ROOT:-/work/hdd/bekh/thayes/openfold_reps_outputs_sm}
export ECOD_PAIR_BLOCKS_ROOT=${ECOD_PAIR_BLOCKS_ROOT:-/work/hdd/bfna/thayes/ecod/experiments/representation_saves}
export ECOD_PDB_ROOT=${ECOD_PDB_ROOT:-/work/hdd/bfna/thayes/ecod/experiments/openfold_reps_outputs_sm}
export DSSP_EXECUTABLE=${DSSP_EXECUTABLE:-$HOME/dssp_env/bin/mkdssp}
export SUBSTRUCTURE_ANALYSIS_SPLITS=${SUBSTRUCTURE_ANALYSIS_SPLITS:-test}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPLCONFIGDIR=${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-local}
mkdir -p "$MPLCONFIGDIR"

ANALYSIS_LAYERS=(0 10 20 30 40 47)
if [[ ${SUBSTRUCTURE_STAGE:-layer} == layer ]]; then
    export SUBSTRUCTURE_LAYER=${ANALYSIS_LAYERS[$SLURM_ARRAY_TASK_ID]}
fi

SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
test -x "$DSSP_EXECUTABLE"
srun python3 -u "$SCRIPT_DIR/diagnostic_secondary_structure_emergence.py"
