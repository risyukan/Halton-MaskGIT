#!/usr/bin/bash
# Wait until all 4 GPUs are basically free, then run the small-384 cfg=1.0
# partial_update sweep: N=1..9 + no-refresh (11 configs), sequentially.
# All params identical to the base-384 cfg=1.0 sweep except the model
# (ImageNet_384_small.pth, --vit-size small).
#
# Output:
#   /tmp/small384_cfg10_sweep_wrapper.log         wrapper-level progress
#   /tmp/eval_fid_small384_cfg10_partialupdate_*.log   per-script stdout (11 files)
#   results/halton_small384_cfg10_partialupdate_sweep.txt   parsed FID/IS rows
#   /tmp/small384_cfg10_sweep_DONE                marker created when all finish
set -uo pipefail

cd /work/q-li/Halton-MaskGIT

WRAPPER_LOG=/tmp/small384_cfg10_sweep_wrapper.log
exec > >(tee -a "$WRAPPER_LOG") 2>&1

echo "============================================================"
echo "[$(date '+%F %T')] small-384 cfg=1.0 sweep wrapper started (PID $$)"
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

# --- Step 2: grace, then run the 11 configs sequentially ----------------
sleep 30
echo "[$(date '+%F %T')] launching 11-config sweep (N1..N9 + norefresh)..."

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

run_one launch/eval_fid_small384_cfg10_partialupdate_N1.sh
run_one launch/eval_fid_small384_cfg10_partialupdate_N2.sh
run_one launch/eval_fid_small384_cfg10_partialupdate_N3.sh
run_one launch/eval_fid_small384_cfg10_partialupdate_N4.sh
run_one launch/eval_fid_small384_cfg10_partialupdate_N5.sh
run_one launch/eval_fid_small384_cfg10_partialupdate_N6.sh
run_one launch/eval_fid_small384_cfg10_partialupdate_N7.sh
run_one launch/eval_fid_small384_cfg10_partialupdate_N8.sh
run_one launch/eval_fid_small384_cfg10_partialupdate_N9.sh
run_one launch/eval_fid_small384_cfg10_partialupdate_norefresh.sh

touch /tmp/small384_cfg10_sweep_DONE
echo "============================================================"
echo "[$(date '+%F %T')] small-384 cfg=1.0 sweep wrapper DONE"
echo "  results -> results/halton_small384_cfg10_partialupdate_sweep.txt"
echo "============================================================"
