#!/bin/bash
#SBATCH --job-name=SaltBridgeLayers
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=02:00:00
#SBATCH --constraint="scratch"
#SBATCH --array=0-47
#SBATCH -o slurm-%A_%a.out
#SBATCH -e slurm-%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ikhire3@gatech.edu

set -euo pipefail

module reset
module load python
source ~/protein_venv/bin/activate

export PAIR_BLOCKS_ROOT=/work/hdd/bekh/thayes/openfold_reps
export CSV_ROOT=$HOME/csv_files/pair_interaction_classification/cameo50_v1
export PROBE_LAYER=$SLURM_ARRAY_TASK_ID
export PROBE_DEADLINE_SECONDS=6000
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}

module list
echo "job is starting on $(hostname) at $(date --iso-8601=seconds)"
echo "probing pair-representation layer $PROBE_LAYER (array task $SLURM_ARRAY_TASK_ID)"

python3 - <<'PY'
import os
from pathlib import Path

import numpy
import scipy
import sklearn
import matplotlib

csv_root = Path(os.environ["CSV_ROOT"])
pair_root = Path(os.environ["PAIR_BLOCKS_ROOT"])
assert pair_root.is_dir(), f"missing pair-block root: {pair_root}"
assert (csv_root / "annotation_config.json").is_file(), "missing annotation_config.json"
assert (csv_root / "protein_manifest.csv").is_file(), "missing protein_manifest.csv"
print(
    "preflight:",
    numpy.__version__,
    scipy.__version__,
    sklearn.__version__,
    matplotlib.__version__,
)
PY

srun python3 -u "$SCRIPT_DIR/diagnostic_salt_bridge_classification_probe.py"
