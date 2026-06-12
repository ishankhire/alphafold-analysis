#!/bin/bash
#SBATCH --job-name=GenTTComb256L0023
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192g
#SBATCH --time=48:00:00
#SBATCH --constraint="scratch"
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ikhire3@gatech.edu

module reset
module load python

source ~/protein_venv/bin/activate

module list
echo "job is starting on $(hostname)"

export CSV_ROOT=$HOME/csv_files

srun python3 -u general_base256_train_test_layers_00_23.py
