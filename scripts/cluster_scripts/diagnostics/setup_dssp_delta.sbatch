#!/bin/bash
#SBATCH --job-name=SetupDSSP
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4g
#SBATCH --time=00:30:00
#SBATCH --constraint="scratch"
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ikhire3@gatech.edu

module reset
module load anaconda3_cpu || module load anaconda3
if [ -d "$HOME/dssp_env" ]; then
    conda install -y -p "$HOME/dssp_env" -c conda-forge dssp=4.6.1
else
    conda create -y -p "$HOME/dssp_env" -c conda-forge dssp=4.6.1
fi
"$HOME/dssp_env/bin/mkdssp" --version
