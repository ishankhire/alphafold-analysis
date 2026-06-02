#!/bin/bash
#SBATCH -JPerProteinRegression           # Job name
#SBATCH -N1 --ntasks-per-node=4          # 1 node, 4 cores (for numpy BLAS threading)
#SBATCH --mem-per-cpu=1G                 # 1 GB/core × 4 = 4 GB total (one protein at a time)
#SBATCH -t18:00:00                       # 18 hour walltime (500 proteins × ~15s each)
#SBATCH -oPerProteinRegression-%j.out    # Output file with job ID
#SBATCH --mail-type=BEGIN,END,FAIL       # Mail on start/end/fail
#SBATCH --mail-user=<YOUR_EMAIL>         # <-- UPDATE THIS

cd $SLURM_SUBMIT_DIR                    # Change to directory where sbatch was run
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

module load anaconda3                    # Load Python environment

# Uncomment and set if $SCRATCH is not pre-configured on the cluster:
# export SCRATCH=/path/to/your/data/root

# Data directories on ICE
export PAIR_BLOCKS_ROOT=$SCRATCH
export PDB_ROOT=$SCRATCH/Structures
export CSV_ROOT=$SCRATCH/csv_files

# NOTE: Using 'python' (not 'srun python') because srun with ntasks=4
# would launch 4 copies of the script. We request 4 cores so numpy BLAS
# can use them for multithreaded linear algebra.
python "$SCRIPT_DIR/../per_protein_regression.py"
