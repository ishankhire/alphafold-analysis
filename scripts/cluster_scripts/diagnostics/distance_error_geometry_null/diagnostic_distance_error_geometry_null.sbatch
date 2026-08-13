#!/bin/bash
#SBATCH --job-name=DistErrGeomNull
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=48:00:00
#SBATCH --array=0-5
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

export INPUT_RESULTS_DIR=$HOME/csv_files/general_base256_combined
export CSV_ROOT=$HOME/csv_files
export CAMEO_PDB_ROOT=/work/hdd/bekh/thayes/openfold_reps_outputs_sm
export ECOD_PDB_ROOT=/work/hdd/bfna/thayes/ecod/experiments/openfold_reps_outputs_sm
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

ANALYSIS_LAYERS=(0 10 20 30 40 47)
export NULL_LAYERS=0,10,20,30,40,47
export NULL_TASK_LAYER=${ANALYSIS_LAYERS[$SLURM_ARRAY_TASK_ID]}

SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
srun python3 -u "$SCRIPT_DIR/diagnostic_distance_error_geometry_null.py"
