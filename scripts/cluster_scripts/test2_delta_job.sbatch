#!/bin/bash
#SBATCH --job-name=Test2MultiProtRegression
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4              # 4 cores for numpy BLAS threading
#SBATCH --mem=32g                      # 32 GB (25 proteins + MDS overhead)
#SBATCH --time=03:00:00                # 3 hour walltime
#SBATCH --constraint="scratch"
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ikhire3@gatech.edu

module reset                           # drop all modules, set $WORK and $SCRATCH
module load python                     # load Python environment

# Activate virtual environment with dependencies
# (one-time setup: python3 -m venv ~/protein_venv && source ~/protein_venv/bin/activate && pip install numpy scipy matplotlib scikit-learn biopython)
source ~/protein_venv/bin/activate

module list                            # job documentation and metadata

echo "job is starting on $(hostname)"

# Data directories on Delta
export PAIR_BLOCKS_ROOT=/work/hdd/bekh/thayes/openfold_reps
export PDB_ROOT=/work/hdd/bekh/thayes/openfold_reps_outputs_sm
export CSV_ROOT=$HOME/linear_regression_csv_files

srun python3 test2_multi_protein_regression.py
