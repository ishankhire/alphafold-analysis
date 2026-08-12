#!/bin/bash
#SBATCH --job-name=AggDistanceControls
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8g
#SBATCH --time=00:10:00
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
export MPLCONFIGDIR=${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}

SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
srun python3 -u "$SCRIPT_DIR/aggregate_local_pair_distance_controls.py"
