#!/bin/bash
#SBATCH --job-name=triangle_layers
#SBATCH --partition=cpu
#SBATCH --array=0-47%12
#SBATCH --cpus-per-task=1
#SBATCH --mem=8g
#SBATCH --time=01:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ikhire3@gatech.edu
set -euo pipefail
SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
source "$HOME/protein_venv/bin/activate"
srun python3 -u "$SCRIPT_DIR/pair_triangle_dominance_layer.py"
