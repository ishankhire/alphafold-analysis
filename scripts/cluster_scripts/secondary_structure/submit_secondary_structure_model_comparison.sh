#!/bin/bash
set -euo pipefail

SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
LAUNCHER="$SCRIPT_DIR/secondary_structure_model_comparison.sbatch"
CONCURRENCY=${SECONDARY_STRUCTURE_ARRAY_CONCURRENCY:-6}
STAMP=$(date +%Y%m%dT%H%M%S)

if ! [[ "$CONCURRENCY" =~ ^[1-9][0-9]*$ ]]; then
  echo "SECONDARY_STRUCTURE_ARRAY_CONCURRENCY must be a positive integer" >&2
  exit 2
fi
for required in secondary_structure_model_comparison.py diagnostic_secondary_structure_emergence.py general_train_test_common.py cluster_regression_helpers.py regression_protocol.py regression_protocol_data/regression_master_split_v1.csv; do
  if [ ! -e "$SCRIPT_DIR/$required" ]; then
    echo "Missing flat-deployment dependency: $SCRIPT_DIR/$required" >&2
    exit 2
  fi
done

prepare_job=$(sbatch --parsable --array=0 --time=24:00:00 --mem=24G --output="secondary_structure_prepare_%A_%a_${STAMP}.out" --error="secondary_structure_prepare_%A_%a_${STAMP}.err" --export=ALL,SECONDARY_STRUCTURE_STAGE=prepare "$LAUNCHER")
layer_job=$(sbatch --parsable --dependency="afterok:$prepare_job" --array="0-47%$CONCURRENCY" --output="secondary_structure_layer_%A_%a_${STAMP}.out" --error="secondary_structure_layer_%A_%a_${STAMP}.err" --export=ALL,SECONDARY_STRUCTURE_STAGE=layer "$LAUNCHER")
consolidate_job=$(sbatch --parsable --dependency="afterok:$layer_job" --array=0 --output="secondary_structure_consolidate_%A_%a_${STAMP}.out" --error="secondary_structure_consolidate_%A_%a_${STAMP}.err" --export=ALL,SECONDARY_STRUCTURE_STAGE=consolidate "$LAUNCHER")

echo "prepare=$prepare_job"
echo "layers=$layer_job"
echo "consolidate=$consolidate_job"
