#!/bin/bash
#SBATCH -JProteinSplitRegression          # Job name
#SBATCH -N1 --ntasks-per-node=4           # 1 node, 4 cores (for numpy BLAS threading)
#SBATCH --mem-per-cpu=16G                 # 16 GB/core × 4 = 64 GB total
#SBATCH -t8:00:00                         # 8 hour walltime (max: 18h)
#SBATCH -oProteinSplitRegression-%j.out   # Output file with job ID
#SBATCH --mail-type=BEGIN,END,FAIL        # Mail on start/end/fail
#SBATCH --mail-user=<YOUR_EMAIL>          # <-- UPDATE THIS

cd $SLURM_SUBMIT_DIR                     # Change to directory where sbatch was run

module load anaconda3                     # Load Python environment

# Uncomment and set if $SCRATCH is not pre-configured on the cluster:
# export SCRATCH=/path/to/your/data/root

# NOTE: Using 'python' (not 'srun python') because srun with ntasks=4
# would launch 4 copies of the script. We request 4 cores so numpy BLAS
# can use them for multithreaded linear algebra.
python train_test_separated_multi_protein_regression.py
