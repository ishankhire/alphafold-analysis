#!/bin/bash
#SBATCH --job-name=PerProteinRegression
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4              # 4 cores for numpy BLAS threading
#SBATCH --mem=4g                       # 4 GB total (one protein at a time)
#SBATCH --time=18:00:00                # 18 hour walltime (max: 48h on cpu partition)
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

# srun with ntasks-per-node=1 launches exactly one copy of the script.
# The 4 cores requested via cpus-per-task are available for numpy BLAS threading.
srun python3 -u per_protein_regression.py
