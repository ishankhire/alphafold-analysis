#!/bin/bash
#SBATCH --job-name=copy_cameo_rescue
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --constraint="scratch&projects&work"
#SBATCH --output=/work/hdd/bfna/ikhire/cameo_copy/logs/copy_cameo_%j.out
#SBATCH --error=/work/hdd/bfna/ikhire/cameo_copy/logs/copy_cameo_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#
# Copy (never move or delete) thayes's CAMEO data and triangle attention maps
# into /work/hdd/bfna/ikhire/thayes/cameo/, mirroring each source root:
#   /work/hdd/bekh/thayes/X  -> work_hdd_bekh/X
#   /work/nvme/bekh/thayes/X -> work_nvme_bekh/X
#   /projects/bekh/thayes/X  -> projects_bekh/X
#   /work/hdd/bfna/X         -> work_hdd_bfna/X
#
# rsync flags: -rlpt keeps mtimes (provenance); --no-owner/--no-group so new
# files take the setgid destination group delta_bfna and are not charged to
# the over-quota bekh group. Re-running is safe: rsync skips finished files.
#
# Submit from /work/hdd/bfna/ikhire/cameo_copy:
#   sbatch --mail-user="${CLUSTER_NOTICE_EMAIL:?Set CLUSTER_NOTICE_EMAIL}" copy_cameo.sbatch
set -u
DST=/work/hdd/bfna/ikhire/thayes/cameo
RUN=/work/hdd/bfna/ikhire/cameo_copy/logs/${SLURM_JOB_ID:-manual}
STREAMS=${STREAMS:-8}
mkdir -p "$DST" "$RUN"
RS="rsync -rlpt --no-owner --no-group --partial"
ts() { date '+%F %T'; }
echo "[$(ts)] start  job=${SLURM_JOB_ID:-manual} streams=$STREAMS"

map_dst() {  # source path -> destination path
  case "$1" in
    /work/hdd/bekh/thayes/*)  echo "$DST/work_hdd_bekh/${1#/work/hdd/bekh/thayes/}" ;;
    /work/nvme/bekh/thayes/*) echo "$DST/work_nvme_bekh/${1#/work/nvme/bekh/thayes/}" ;;
    /work/nvme/bekh/*)        echo "$DST/work_nvme_bekh/${1#/work/nvme/bekh/}" ;;
    /projects/bekh/thayes/*)  echo "$DST/projects_bekh/${1#/projects/bekh/thayes/}" ;;
    /work/hdd/bfna/*)         echo "$DST/work_hdd_bfna/${1#/work/hdd/bfna/}" ;;
    *) echo "UNMAPPED"; return 1 ;;
  esac
}

# Tier 1: MSAs, reference structures, predicted and ablation PDBs (~30 GiB).
TIER1=(
  /projects/bekh/thayes/alignments
  /projects/bekh/thayes/cameo_dset
  /work/hdd/bekh/thayes/openfold_reps_outputs_sm
  /work/hdd/bekh/thayes/openfold_reps_outputs
  /work/nvme/bekh/thayes/cameo_reps_outputs
  /work/hdd/bfna/openfold_outputs_layerwise_ablation
  /work/hdd/bfna/openfold_outputs_global_ablations
  /work/hdd/bfna/openfold_outputs_fixed_ablation
  /work/nvme/bekh/openfold_outputs_layerwise_ablation
  /work/hdd/bfna/gprash/cameo_pdbs
)
# Tier 2: 48-layer pair + MSA blocks (386 GiB), split per protein.
# Tier 3: triangle attention maps (347 GiB), large dirs split per child.
SPLIT=( /work/hdd/bekh/thayes/openfold_reps /work/hdd/bekh/thayes/cameo_triplets /work/nvme/bekh/thayes/cameo_triplets )
TIER3_WHOLE=(
  /work/hdd/bfna/attention_map_dir_test_triplets
  /work/hdd/bekh/thayes/test_attn
  /work/hdd/bekh/thayes/temp_store
)

TASKS="$RUN/tasks.txt"; : > "$TASKS"
printf '%s\n' "${TIER1[@]}" >> "$TASKS"
for d in "${SPLIT[@]}"; do
  find "$d" -mindepth 1 -maxdepth 1 | sort >> "$TASKS"
done
printf '%s\n' "${TIER3_WHOLE[@]}" >> "$TASKS"
echo "[$(ts)] $(wc -l < "$TASKS") copy tasks"

copy_one() {
  src=$1; dst=$(map_dst "$src") || { echo "UNMAPPED $src"; return 1; }
  mkdir -p "$(dirname "$dst")"
  t0=$(date +%s)
  if [ -d "$src" ]; then $RS "$src/" "$dst/"; else $RS "$src" "$dst"; fi
  rc=$?
  printf '%s\t%s\t%s\t%ss\t%s\n' "$(ts)" "$rc" "$src" "$(( $(date +%s) - t0 ))" "$dst"
}
export -f copy_one map_dst ts; export DST RS
xargs -d '\n' -P "$STREAMS" -I{} bash -c 'copy_one "$1"' _ {} < "$TASKS" \
  > "$RUN/copy_results.tsv" 2> "$RUN/rsync_errors.txt"
echo "[$(ts)] copy done; nonzero rsync exits: $(awk -F'\t' '$2!=0' "$RUN/copy_results.tsv" | wc -l)"

# Verification: a dry-run rsync must report no pending transfers, and file
# counts / apparent bytes must match per task.
verify_one() {
  src=$1; dst=$(map_dst "$src")
  if [ -d "$src" ]; then s="$src/"; d="$dst/"; else s="$src"; d="$dst"; fi
  pending=$($RS --dry-run --itemize-changes "$s" "$d" 2>/dev/null | grep -c '^[<>c]')
  sf=$(find "$src" -type f 2>/dev/null | wc -l); df=$(find "$dst" -type f 2>/dev/null | wc -l)
  sb=$(du -sb "$src" 2>/dev/null | cut -f1);     db=$(du -sb "$dst" 2>/dev/null | cut -f1)
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$src" "$pending" "$sf" "$df" "$sb" "$db"
}
export -f verify_one
printf 'source\tpending\tsrc_files\tdst_files\tsrc_bytes\tdst_bytes\n' > "$RUN/verify.tsv"
xargs -d '\n' -P "$STREAMS" -I{} bash -c 'verify_one "$1"' _ {} < "$TASKS" >> "$RUN/verify.tsv"
echo "[$(ts)] verify done; tasks with mismatches: $(awk -F'\t' 'NR>1 && ($2!=0 || $3!=$4)' "$RUN/verify.tsv" | wc -l)"
