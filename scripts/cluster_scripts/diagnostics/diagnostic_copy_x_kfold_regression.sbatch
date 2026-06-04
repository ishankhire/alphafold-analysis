#!/bin/bash
#SBATCH --job-name=DiagCopyXKFold
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH --time=12:00:00
#SBATCH --constraint="scratch"
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ikhire3@gatech.edu

module reset
module load python

# Activate virtual environment with dependencies
source ~/protein_venv/bin/activate

module list
echo "job is starting on $(hostname)"

if [[ -f "$SLURM_SUBMIT_DIR/diagnostic_copy_x_kfold_regression.py" ]]; then
    cd "$SLURM_SUBMIT_DIR"
elif [[ -f "$SLURM_SUBMIT_DIR/scripts/cluster_scripts/diagnostics/diagnostic_copy_x_kfold_regression.py" ]]; then
    cd "$SLURM_SUBMIT_DIR/scripts/cluster_scripts/diagnostics"
else
    echo "Could not find diagnostic_copy_x_kfold_regression.py from SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR" >&2
    exit 1
fi

# Data directories on Delta
export PAIR_BLOCKS_ROOT=/work/hdd/bekh/thayes/openfold_reps
export PDB_ROOT=/work/hdd/bekh/thayes/openfold_reps_outputs_sm
export CSV_ROOT=$HOME/diagnostic_copy_x_kfold_csv_files

srun python3 -u diagnostic_copy_x_kfold_regression.py
