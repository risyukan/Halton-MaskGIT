#!/usr/bin/bash
# Wait until all 4 GPUs are basically free, then run base-384 cfg=0.7
# partial_update sweep for N=7,8,9 sequentially (N5/N6 already done).
# All params identical to the existing N1..N6 scripts except HALTON_CACHE_REFRESH_N.
#
# Output:
#   /tmp/base384_cfg07_sweep_N7toN9_wrapper.log   wrapper-level progress
#   /tmp/eval_fid_base384_cfg07_partialupdate_N{7,8,9}.log   per-script stdout
#   results/halton_base384_cfg07_partialupdate_sweep.txt   parsed FID/IS rows
#   /tmp/base384_cfg07_sweep_N7toN9_DONE          marker created when all finish
set -uo pipefail

cd /work/q-li/Halton-MaskGIT

WRAPPER_LOG=/tmp/base384_cfg07_sweep_N7toN9_wrapper.log
exec > >(tee -a "$WRAPPER_LOG") 2>&1

echo "============================================================"
echo "[$(date '+%F %T')] base-384 cfg=0.7 sweep N7..N9 wrapper started (PID $$)"
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

# --- Step 2: grace, then run N7,N8,N9 sequentially ----------------------
sleep 30
echo "[$(date '+%F %T')] launching N7..N9..."

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

run_one launch/eval_fid_base384_cfg07_partialupdate_N7.sh
run_one launch/eval_fid_base384_cfg07_partialupdate_N8.sh
run_one launch/eval_fid_base384_cfg07_partialupdate_N9.sh

touch /tmp/base384_cfg07_sweep_N7toN9_DONE
echo "============================================================"
echo "[$(date '+%F %T')] base-384 cfg=0.7 sweep N7..N9 wrapper DONE"
echo "  results -> results/halton_base384_cfg07_partialupdate_sweep.txt"
echo "============================================================"
