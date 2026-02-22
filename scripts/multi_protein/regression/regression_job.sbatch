#!/bin/bash
#SBATCH -JMultiProteinRegression          # Job name
#SBATCH -N1 --ntasks-per-node=1           # 1 node, 1 task (run script once)
#SBATCH --mem-per-cpu=8G                  # 8 GB memory
#SBATCH -t2:00:00                         # 2 hour walltime
#SBATCH -oRegression-Report-%j.out        # Output file with job ID
#SBATCH --mail-type=BEGIN,END,FAIL        # Mail on start/end/fail
#SBATCH --mail-user=<YOUR_EMAIL>          # <-- UPDATE THIS

cd $SLURM_SUBMIT_DIR                     # Change to directory where sbatch was run

module load anaconda3                     # Load Python environment

# Uncomment and set if $SCRATCH is not pre-configured on the cluster:
# export SCRATCH=/path/to/your/data/root

srun python cluster_multi_protein_regression.py
