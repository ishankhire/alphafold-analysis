#!/bin/bash
#SBATCH --job-name=MultiProteinRegression
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4              # 4 cores for numpy BLAS threading
#SBATCH --mem=64g                      # 64 GB total
#SBATCH --time=12:00:00                # 8 hour walltime (max: 48h on cpu partition)
#SBATCH --constraint="scratch"
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ikhire3@gatech.edu

module reset                           # drop all modules, set $WORK and $SCRATCH
module load python                     # load Python environment
module list                            # job documentation and metadata

echo "job is starting on $(hostname)"

# Data directories on Delta
export PAIR_BLOCKS_ROOT=/work/hdd/bekh/thayes/openfold_reps
export PDB_ROOT=/work/hdd/bekh/thayes/openfold_reps_outputs_sm
export CSV_ROOT=/work/hdd/bekh/thayes/linear_regression_csv_files

# srun with ntasks-per-node=1 launches exactly one copy of the script.
# The 4 cores requested via cpus-per-task are available for numpy BLAS threading.
srun python3 final_cluster_multi_protein_regression.py
