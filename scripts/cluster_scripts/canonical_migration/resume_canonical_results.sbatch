#!/bin/bash
#SBATCH --job-name=CanonicalResume
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6g
#SBATCH --time=04:00:00
#SBATCH --constraint="scratch&work"
#SBATCH -o canonical_resume_%j.out
#SBATCH -e canonical_resume_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ikhire3@gatech.edu

set -euo pipefail
module reset
module load python
source ~/protein_venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
srun python3 -u "$SCRIPT_DIR/migrate_canonical_results.py" --execute --resume-published \
    --authority "$SCRIPT_DIR/migration_inputs/authority.json" \
    --replacements "$SCRIPT_DIR/migration_inputs/refreshed"
