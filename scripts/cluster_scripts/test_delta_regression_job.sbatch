#!/bin/bash
#SBATCH --job-name=TestMultiProtRegression
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2              # 2 cores (lighter test job)
#SBATCH --mem=16g                      # 16 GB (10 proteins need much less)
#SBATCH --time=01:00:00                # 1 hour walltime
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
export CSV_ROOT=/work/hdd/bekh/thayes/linear_regression_csv_files

srun python3 test_cluster_multi_protein_regression.py
