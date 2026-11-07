#!/bin/bash
#SBATCH --job-name=TransferFollow
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2g
#SBATCH --time=00:15:00
#SBATCH --constraint="scratch&work"
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# Writes one text report per analysis, plus the tables and chart rows. Needs both replay
# arrays and the ablation CSVs. Pass the notice recipient at submission; see the README.
module reset
module load python
source ~/protein_venv/bin/activate

export CSV_ROOT=${CSV_ROOT:-/work/hdd/bfna/ikhire}
export LAYERWISE_ABLATION_CSV_DIR=${LAYERWISE_ABLATION_CSV_DIR:-/work/hdd/bfna/toDelete/thayes/FINAL_RESULT_CSVS_TRI_END_FIX_PATCHED_FROM_OLD}
export TRANSFER_FOLLOWUPS_REPORT_DIR=${TRANSFER_FOLLOWUPS_REPORT_DIR:-$HOME/diagnostic_experiments/transfer_followups_cameo_v1}
SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
python3 "$SCRIPT_DIR/canonical_storage.py" --check-write "$TRANSFER_FOLLOWUPS_REPORT_DIR" || exit $?
mkdir -p "$TRANSFER_FOLLOWUPS_REPORT_DIR"
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

run() { echo "[$(date -u +%FT%TZ)] stage $1"; srun python3 -u "$SCRIPT_DIR/$1.py" > "$TRANSFER_FOLLOWUPS_REPORT_DIR/$2" || { echo "failed $1"; exit 1; }; }
run analyze_transfer_breakdown transfer_breakdown_report.txt
run analyze_ablation ablation_report.txt
run analyze_gain_position gain_and_chain_report.txt
run analyze_triangle_strata triangle_strata_report.txt
run analyze_violation_events violation_events_report.txt
run export_report_data export_report_data.txt
echo "[$(date -u +%FT%TZ)] end"
