#!/bin/bash
set -euo pipefail

# Validate cached inputs, run all six sparse layer decoders concurrently, then
# consolidate trajectories and plots. Submit this file from any directory.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SBATCH_FILE="$SCRIPT_DIR/diagnostic_secondary_structure_sequence_gated.sbatch"
cd "$SCRIPT_DIR"

PREP_JOB_ID=$(
    sbatch --parsable --array=0 \
        --job-name=SeqGateSecPrep \
        --export=ALL,SUBSTRUCTURE_STAGE=prepare \
        "$SBATCH_FILE"
)
LAYER_JOB_ID=$(
    sbatch --parsable --array=0-5%6 \
        --job-name=SeqGateSecLayers \
        --dependency="afterok:$PREP_JOB_ID" \
        --export=ALL,SUBSTRUCTURE_STAGE=layer \
        "$SBATCH_FILE"
)
CONSOLIDATE_JOB_ID=$(
    sbatch --parsable --array=0 \
        --job-name=SeqGateSecMerge \
        --dependency="afterok:$LAYER_JOB_ID" \
        --export=ALL,SUBSTRUCTURE_STAGE=consolidate \
        "$SBATCH_FILE"
)

echo "preparation validation job: $PREP_JOB_ID"
echo "six-layer array job:       $LAYER_JOB_ID"
echo "consolidation job:         $CONSOLIDATE_JOB_ID"
