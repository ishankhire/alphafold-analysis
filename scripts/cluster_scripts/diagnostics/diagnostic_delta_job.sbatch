#!/bin/bash
#SBATCH --job-name=DiagnosticRegression
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4              # 4 cores for numpy BLAS threading
#SBATCH --mem=32g                      # 32 GB (100 proteins, <=500 residues, 1 layer)
#SBATCH --time=02:00:00                # 2 hours (layer 47 only, 3 sample runs)
#SBATCH --constraint="scratch"
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ikhire3@gatech.edu

module reset                           # drop all modules, set $WORK and $SCRATCH
module load python                     # load Python environment

# Activate virtual environment with dependencies
source ~/protein_venv/bin/activate

module list                            # job documentation and metadata

echo "job is starting on $(hostname)"

# Data directories on Delta
export PAIR_BLOCKS_ROOT=/work/hdd/bekh/thayes/openfold_reps
export PDB_ROOT=/work/hdd/bekh/thayes/openfold_reps_outputs_sm
export CSV_ROOT=$HOME/diagnostic_csv_files

srun python3 -u diagnostic_regression.py
