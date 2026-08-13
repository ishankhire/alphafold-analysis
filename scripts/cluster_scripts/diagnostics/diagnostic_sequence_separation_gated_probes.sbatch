#!/bin/bash
#SBATCH --job-name=DiagSeqSepGated
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192g
#SBATCH --time=24:00:00
#SBATCH --constraint="scratch"
#SBATCH --array=0-5
#SBATCH -o slurm-%A_%a.out
#SBATCH -e slurm-%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ikhire3@gatech.edu

module reset
module load python

source ~/protein_venv/bin/activate

module list
echo "job is starting on $(hostname)"
echo "array task ${SLURM_ARRAY_TASK_ID}"

export CSV_ROOT=${CSV_ROOT:-$HOME/csv_files}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
srun python3 -u "$SCRIPT_DIR/diagnostic_sequence_separation_gated_probes.py"
