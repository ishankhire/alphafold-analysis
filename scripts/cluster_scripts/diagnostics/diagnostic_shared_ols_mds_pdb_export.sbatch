#!/bin/bash
#SBATCH --job-name=SharedOLSMdsPDB
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8g
#SBATCH --time=00:30:00
#SBATCH --constraint="scratch&work"
#SBATCH -o shared_ols_mds_pdb_%j.out
#SBATCH -e shared_ols_mds_pdb_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ikhire3@gatech.edu

module reset
module load python
source ~/protein_venv/bin/activate
module list

export CSV_ROOT=${CSV_ROOT:-$HOME/csv_files}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

export MDS_PDB_OUTPUT_ROOT=${MDS_PDB_OUTPUT_ROOT:-$CSV_ROOT/regression_5fold_v1/analysis/shared_ols_mds_pdb_exports}
LOG_DIR=$MDS_PDB_OUTPUT_ROOT/logs
mkdir -p "$LOG_DIR"
RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
exec > >(tee -a "$LOG_DIR/run_${RUN_TIMESTAMP}_${SLURM_JOB_ID}.log") 2>&1

echo "[$(date --iso-8601=seconds)] shared-OLS MDS PDB export starting on $(hostname)"
SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
srun python3 -u "$SCRIPT_DIR/diagnostic_shared_ols_mds_pdb_export.py"
