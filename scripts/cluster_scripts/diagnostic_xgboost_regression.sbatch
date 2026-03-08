#!/bin/bash
#SBATCH --job-name=DiagXGBoost
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48g
#SBATCH --time=02:30:00
#SBATCH --constraint="scratch"
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ikhire3@gatech.edu

module reset
module load python

# Activate virtual environment with dependencies
# One-time setup if needed in the venv: pip install xgboost
source ~/protein_venv/bin/activate

module list
echo "job is starting on $(hostname)"

export PAIR_BLOCKS_ROOT=/work/hdd/bekh/thayes/openfold_reps
export PDB_ROOT=/work/hdd/bekh/thayes/openfold_reps_outputs_sm
export CSV_ROOT=$HOME/diagnostic_xgboost_csv_files

srun python3 -u diagnostic_xgboost_regression.py
