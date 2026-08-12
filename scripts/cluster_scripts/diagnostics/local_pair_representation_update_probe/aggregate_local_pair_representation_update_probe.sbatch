#!/bin/bash
#SBATCH --job-name=AggLocalPairRep
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12g
#SBATCH --time=00:30:00
#SBATCH --constraint="scratch"
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ikhire3@gatech.edu

module reset
module load python
source ~/protein_venv/bin/activate
module list

echo "aggregation job is starting on $(hostname)"
export CSV_ROOT=$HOME/csv_files

SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
srun python3 -u "$SCRIPT_DIR/aggregate_local_pair_representation_update_probe.py"
