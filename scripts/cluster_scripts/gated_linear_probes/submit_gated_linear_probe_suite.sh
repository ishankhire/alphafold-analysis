#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
RUN_MODE=${GATED_PROBE_RUN_MODE:-production}
if [[ "$RUN_MODE" != "production" && "$RUN_MODE" != "smoke" ]]; then
    echo "GATED_PROBE_RUN_MODE must be production or smoke" >&2
    exit 2
fi
export GATED_PROBE_RUN_MODE=$RUN_MODE

prepare_job=$(sbatch --parsable "$SCRIPT_DIR/prepare_gated_linear_probe_suite.sbatch")
if [[ "$RUN_MODE" == "smoke" ]]; then
    sequence_job=$(sbatch --parsable --array=0-0 --dependency="afterok:$prepare_job" "$SCRIPT_DIR/sequence_gated_linear_probes.sbatch")
    aa_job=$(sbatch --parsable --array=0-0 --dependency="afterok:$prepare_job" "$SCRIPT_DIR/aa_pair_gated_linear_probes.sbatch")
    joint_job=$(sbatch --parsable --array=0-5%6 --dependency="afterok:$prepare_job" "$SCRIPT_DIR/aa_sequence_gated_linear_probes.sbatch")
    joint_layer_job=$(sbatch --parsable --array=0-0 --dependency="afterok:$joint_job" "$SCRIPT_DIR/consolidate_joint_gated_layer.sbatch")
    output_root="${CSV_ROOT:-$HOME/csv_files}/gated_linear_probes_smoke"
else
    sequence_job=$(sbatch --parsable --dependency="afterok:$prepare_job" "$SCRIPT_DIR/sequence_gated_linear_probes.sbatch")
    aa_job=$(sbatch --parsable --dependency="afterok:$prepare_job" "$SCRIPT_DIR/aa_pair_gated_linear_probes.sbatch")
    joint_job=$(sbatch --parsable --dependency="afterok:$prepare_job" "$SCRIPT_DIR/aa_sequence_gated_linear_probes.sbatch")
    joint_layer_job=$(sbatch --parsable --dependency="afterok:$joint_job" "$SCRIPT_DIR/consolidate_joint_gated_layer.sbatch")
    output_root="${CSV_ROOT:-$HOME/csv_files}/gated_linear_probes"
fi
aggregate_job=$(sbatch --parsable --dependency="afterok:$sequence_job:$aa_job:$joint_layer_job" "$SCRIPT_DIR/aggregate_gated_linear_probe_suite.sbatch")

echo "run mode:       $RUN_MODE"
echo "preparation:    $prepare_job"
echo "sequence array: $sequence_job"
echo "AA array:       $aa_job"
echo "joint array:    $joint_job"
echo "joint layers:   $joint_layer_job"
echo "aggregation:    $aggregate_job"
echo "output root:    $output_root"
