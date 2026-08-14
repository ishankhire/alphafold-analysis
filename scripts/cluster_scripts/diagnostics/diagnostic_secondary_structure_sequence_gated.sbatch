#!/bin/bash
#SBATCH --job-name=SeqGateSecStruct
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8g
#SBATCH --time=00:25:00
#SBATCH --array=0-5%6
#SBATCH --constraint="work"
#SBATCH --requeue
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

export CSV_ROOT=${CSV_ROOT:-$HOME/csv_files}
export SEQUENCE_GATED_SUITE_DIR=${SEQUENCE_GATED_SUITE_DIR:-$CSV_ROOT/gated_linear_probes/consolidated}
export SEQUENCE_GATED_RESULTS_DIR=${SEQUENCE_GATED_RESULTS_DIR:-$SEQUENCE_GATED_SUITE_DIR/analysis_1_sequence_gated}
export SUBSTRUCTURE_REFERENCE_DIR=${SUBSTRUCTURE_REFERENCE_DIR:-$CSV_ROOT/diagnostic_secondary_structure_emergence}
export CAMEO_PAIR_BLOCKS_ROOT=${CAMEO_PAIR_BLOCKS_ROOT:-/work/hdd/bekh/thayes/openfold_reps}
export CAMEO_PDB_ROOT=${CAMEO_PDB_ROOT:-/work/hdd/bekh/thayes/openfold_reps_outputs_sm}
export ECOD_PAIR_BLOCKS_ROOT=${ECOD_PAIR_BLOCKS_ROOT:-/work/hdd/bfna/thayes/ecod/experiments/representation_saves}
export ECOD_PDB_ROOT=${ECOD_PDB_ROOT:-/work/hdd/bfna/thayes/ecod/experiments/openfold_reps_outputs_sm}
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
srun python3 -u "$SCRIPT_DIR/diagnostic_secondary_structure_sequence_gated.py"
