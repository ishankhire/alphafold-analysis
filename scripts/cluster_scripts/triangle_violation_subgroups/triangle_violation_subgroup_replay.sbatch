#!/bin/bash
#SBATCH --job-name=TriSubReplay
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8g
#SBATCH --time=06:00:00
#SBATCH --constraint="work"
#SBATCH --array=0-47%12
#SBATCH -o slurm-%A_%a.out
#SBATCH -e slurm-%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ikhire3@gatech.edu

module reset
module load python
source ~/protein_venv/bin/activate

export CSV_ROOT=$HOME/csv_files
export TRIANGLE_PER_PROTEIN_OLS_ROOT=$CSV_ROOT/per_protein_base256_combined
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
srun python3 -u "$SCRIPT_DIR/triangle_violation_subgroup_replay.py"
