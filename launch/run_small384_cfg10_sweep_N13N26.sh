#!/usr/bin/bash
# Wait until all 4 GPUs are basically free, then run ONLY the two新增 points
# that fill the FID-cliff gap between N=9 (3 refresh) and ∞ (0 refresh):
#   N=13 -> 2/26 refresh,  N=26 -> 1/26 refresh.
# Appends to the same results/halton_small384_cfg10_partialupdate_sweep.txt.
#
# Output:
#   /tmp/small384_cfg10_sweep_N13N26_wrapper.log   wrapper-level progress
#   /tmp/eval_fid_small384_cfg10_partialupdate_N13.log / _N26.log
#   /tmp/small384_cfg10_sweep_N13N26_DONE          marker created when finished
set -uo pipefail

cd /work/q-li/Halton-MaskGIT

WRAPPER_LOG=/tmp/small384_cfg10_sweep_N13N26_wrapper.log
exec > >(tee -a "$WRAPPER_LOG") 2>&1

echo "============================================================"
echo "[$(date '+%F %T')] small-384 cfg=1.0 N13/N26 wrapper started (PID $$)"
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

# --- Step 2: grace, then run the 2 configs sequentially -----------------
sleep 30
echo "[$(date '+%F %T')] launching 2-config gap-fill (N13, N26)..."

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

run_one launch/eval_fid_small384_cfg10_partialupdate_N13.sh
run_one launch/eval_fid_small384_cfg10_partialupdate_N26.sh

touch /tmp/small384_cfg10_sweep_N13N26_DONE
echo "============================================================"
echo "[$(date '+%F %T')] small-384 cfg=1.0 N13/N26 wrapper DONE"
echo "  results -> results/halton_small384_cfg10_partialupdate_sweep.txt"
echo "============================================================"
