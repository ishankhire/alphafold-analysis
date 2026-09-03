#!/usr/bin/env bash

set -euo pipefail

readonly SOURCE_DIR="${LAYERWISE_ABLATION_SOURCE_DIR:-/work/hdd/bfna/thayes/FINAL_RESULTS_CSVS_TRI_END_FIX_PATCHED_FROM_OLD}"
readonly CSV_ROOT="${CSV_ROOT:-${HOME}/csv_files}"
readonly DEST_DIR="${CSV_ROOT}/layerwise_ablation_csvs"

readonly -a CSV_FILES=(
    "ablate_msa_col_layerwise_fwd.csv"
    "ablate_msa_col_layerwise_rev.csv"
    "ablate_msa_row_layerwise_fwd.csv"
    "ablate_msa_row_layerwise_rev.csv"
    "ablate_msa_transition_layerwise_fwd.csv"
    "ablate_msa_transition_layerwise_rev.csv"
    "ablate_pair_transition_layerwise_fwd.csv"
    "ablate_pair_transition_layerwise_rev.csv"
    "ablate_tmu_layerwise_fwd.csv"
    "ablate_tmu_layerwise_rev.csv"
    "ablate_tri_start_ablate_tri_end_ablate_tmu_layerwise_fwd.csv"
    "ablate_tri_start_ablate_tri_end_ablate_tmu_layerwise_rev.csv"
    "ablate_tri_start_ablate_tri_end_layerwise_fwd.csv"
    "ablate_tri_start_ablate_tri_end_layerwise_rev.csv"
    "structure_module_tm_scores.csv"
)

missing_files=()
for filename in "${CSV_FILES[@]}"; do
    source_path="${SOURCE_DIR}/${filename}"
    if [[ ! -f "${source_path}" || ! -r "${source_path}" ]]; then
        missing_files+=("${source_path}")
    fi
done

if (( ${#missing_files[@]} > 0 )); then
    echo "ERROR: No files were copied because these sources are missing or unreadable:" >&2
    printf '  %s\n' "${missing_files[@]}" >&2
    exit 1
fi

mkdir -p "${DEST_DIR}"

for filename in "${CSV_FILES[@]}"; do
    source_path="${SOURCE_DIR}/${filename}"
    destination_path="${DEST_DIR}/${filename}"

    cp -- "${source_path}" "${destination_path}"
    if ! cmp -s -- "${source_path}" "${destination_path}"; then
        echo "ERROR: Copied file does not match its source: ${filename}" >&2
        exit 1
    fi
    echo "Copied and verified: ${filename}"
done

echo "Copied ${#CSV_FILES[@]} unchanged CSV files to ${DEST_DIR}"
