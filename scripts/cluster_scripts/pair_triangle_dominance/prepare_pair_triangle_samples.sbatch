#!/bin/bash
#SBATCH --job-name=triangle_samples
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4g
#SBATCH --time=00:30:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ikhire3@gatech.edu
set -euo pipefail
SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
source "$HOME/protein_venv/bin/activate"
srun python3 -u "$SCRIPT_DIR/prepare_pair_triangle_samples.py"
