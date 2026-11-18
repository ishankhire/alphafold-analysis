#!/bin/bash
#SBATCH --job-name=PerChannelMDSAgg
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8g
#SBATCH --time=00:15:00
#SBATCH --constraint="scratch&work"
#SBATCH -o /work/hdd/bfna/ikhire/diagnostic_experiments/per_channel_mds_cameo_v1/logs/aggregate_per_channel_mds_%j.out
#SBATCH -e /work/hdd/bfna/ikhire/diagnostic_experiments/per_channel_mds_cameo_v1/logs/aggregate_per_channel_mds_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# Collects the layer shards into tables/ after the array (submit with
# --dependency=afterok:<array job>). Refuses to run unless every layer validated.
module reset
module load python
source ~/protein_venv/bin/activate

export CSV_ROOT=${CSV_ROOT:-/work/hdd/bfna/ikhire}
export PER_CHANNEL_MDS_OUTPUT_ROOT=${PER_CHANNEL_MDS_OUTPUT_ROOT:-$CSV_ROOT/diagnostic_experiments/per_channel_mds_cameo_v1}
SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
python3 "$SCRIPT_DIR/canonical_storage.py" --check-write "$PER_CHANNEL_MDS_OUTPUT_ROOT" || exit $?
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "[$(date -u +%FT%TZ)] start aggregation output=$PER_CHANNEL_MDS_OUTPUT_ROOT"
srun python3 -u "$SCRIPT_DIR/aggregate_per_channel_mds.py"
status=$?
echo "[$(date -u +%FT%TZ)] end aggregation status=$status"
exit $status
