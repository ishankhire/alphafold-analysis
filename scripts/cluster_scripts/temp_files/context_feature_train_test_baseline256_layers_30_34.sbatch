#!/bin/bash
#SBATCH --job-name=CtxTT256L3034
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=03:00:00
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

export PAIR_BLOCKS_ROOT=/work/hdd/bekh/thayes/openfold_reps
export PDB_ROOT=/work/hdd/bekh/thayes/openfold_reps_outputs_sm
export CSV_ROOT=$HOME/context_feature_kfold_csv_files

srun python3 -u context_feature_train_test_baseline256_layers_30_34.py
