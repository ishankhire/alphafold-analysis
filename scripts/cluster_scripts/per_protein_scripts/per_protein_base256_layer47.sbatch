#!/bin/bash
#SBATCH --job-name=PerProtB256L47
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8g
#SBATCH --time=18:00:00
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

if [[ -f "$SLURM_SUBMIT_DIR/per_protein_base256_layer47.py" ]]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
elif [[ -f "$SLURM_SUBMIT_DIR/scripts/cluster_scripts/per_protein_scripts/per_protein_base256_layer47.py" ]]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR/scripts/cluster_scripts/per_protein_scripts"
else
    echo "Could not locate per_protein_base256_layer47.py from SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR" >&2
    echo "Submit this job from the repo root or scripts/cluster_scripts/per_protein_scripts." >&2
    exit 1
fi

cd "$SCRIPT_DIR"

export CSV_ROOT=$HOME/csv_files
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun python3 -u per_protein_base256_layer47.py
