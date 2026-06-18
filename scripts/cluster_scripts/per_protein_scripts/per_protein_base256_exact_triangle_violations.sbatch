#!/bin/bash
#SBATCH --job-name=PerProt256ExactTri
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
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

module list
echo "job is starting on $(hostname)"
echo "array job id: ${SLURM_ARRAY_JOB_ID:-none}"
echo "array task id/layer: ${SLURM_ARRAY_TASK_ID:-none}"

export CSV_ROOT=$HOME/csv_files
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
srun python3 -u "$SCRIPT_DIR/per_protein_base256_exact_triangle_violations.py"
