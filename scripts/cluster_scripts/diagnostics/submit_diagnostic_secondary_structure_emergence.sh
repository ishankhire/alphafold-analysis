#!/bin/bash
set -euo pipefail

# Submit one preparation task, the dependency-gated six-layer array, and one
# final consolidation/plotting task. Submit this file from any directory.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SBATCH_FILE="$SCRIPT_DIR/diagnostic_secondary_structure_emergence.sbatch"
ARRAY_CONCURRENCY=${SUBSTRUCTURE_ARRAY_CONCURRENCY:-3}
cd "$SCRIPT_DIR"

PREP_JOB_ID=$(
    sbatch --parsable --array=0 \
        --job-name=SecStructPrep \
        --export=ALL,SUBSTRUCTURE_STAGE=prepare \
        "$SBATCH_FILE"
)
LAYER_JOB_ID=$(
    sbatch --parsable --array="0-5%${ARRAY_CONCURRENCY}" \
        --job-name=SecStructLayers \
        --dependency="afterok:$PREP_JOB_ID" \
        --export=ALL,SUBSTRUCTURE_STAGE=layer \
        "$SBATCH_FILE"
)
CONSOLIDATE_JOB_ID=$(
    sbatch --parsable --array=0 \
        --job-name=SecStructMerge \
        --dependency="afterok:$LAYER_JOB_ID" \
        --export=ALL,SUBSTRUCTURE_STAGE=consolidate \
        "$SBATCH_FILE"
)

echo "preparation job:  $PREP_JOB_ID"
echo "layer array job:  $LAYER_JOB_ID"
echo "consolidation job: $CONSOLIDATE_JOB_ID"
