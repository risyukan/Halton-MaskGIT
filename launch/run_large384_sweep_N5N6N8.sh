#!/usr/bin/bash
# Wait until all 4 GPUs are basically free, then run large-384 cfg=0.5
# partial_update sweep for N=5,6,8 sequentially (fills the N4->∞ gap).
# All params identical to the existing N2/N3/N4 scripts except HALTON_CACHE_REFRESH_N.
#
# Output:
#   /tmp/large384_sweep_N5N6N8_wrapper.log   wrapper-level progress
#   /tmp/eval_fid_partialupdate_N{5,6,8}.log per-script stdout
#   results/halton_large384_partialupdate_sweep.txt   parsed FID/IS rows
#   /tmp/large384_sweep_N5N6N8_DONE          marker created when all finish
set -uo pipefail

cd /work/q-li/Halton-MaskGIT

WRAPPER_LOG=/tmp/large384_sweep_N5N6N8_wrapper.log
exec > >(tee -a "$WRAPPER_LOG") 2>&1

echo "============================================================"
echo "[$(date '+%F %T')] large-384 cfg=0.5 sweep N5/N6/N8 wrapper started (PID $$)"
echo "============================================================"

# --- Step 1: wait until all 4 GPUs are basically free (<2000 MiB) --------
echo "[$(date '+%F %T')] polling nvidia-smi for free GPUs (<2000 MiB each)..."
while true; do
  MAX_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
              | tr -d ' ' | sort -n | tail -1)
  if [ -z "$MAX_USED" ]; then
    echo "[$(date '+%F %T')] nvidia-smi parse failed, retrying in 60s..."
    sleep 60
    continue
  fi
  if [ "$MAX_USED" -lt 2000 ]; then
    echo "[$(date '+%F %T')] all GPUs free (max used = ${MAX_USED} MiB)."
    break
  fi
  echo "[$(date '+%F %T')] GPUs still busy (max ${MAX_USED} MiB), wait 120s..."
  sleep 120
done

# --- Step 2: grace, then run N5,N6,N8 sequentially ----------------------
sleep 30
echo "[$(date '+%F %T')] launching N5/N6/N8..."

run_one () {
  local script="$1"
  echo "------------------------------------------------------------"
  echo "[$(date '+%F %T')] >>> bash $script"
  if bash "$script"; then
    echo "[$(date '+%F %T')] <<< OK   $script"
  else
    rc=$?
    echo "[$(date '+%F %T')] <<< FAIL rc=$rc  $script  (continuing with next)"
  fi
}

run_one launch/eval_fid_large384_partialupdate_N5.sh
run_one launch/eval_fid_large384_partialupdate_N6.sh
run_one launch/eval_fid_large384_partialupdate_N8.sh

touch /tmp/large384_sweep_N5N6N8_DONE
echo "============================================================"
echo "[$(date '+%F %T')] large-384 cfg=0.5 sweep N5/N6/N8 wrapper DONE"
echo "  results -> results/halton_large384_partialupdate_sweep.txt"
echo "============================================================"
