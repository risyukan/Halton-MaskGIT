#!/usr/bin/bash
# Wait until other users' GPU jobs finish, then run base-384 cfg=0.7 sweep
# of 5 partial_update configs (N=1/2/3/4 + no-refresh). Self-monitored.
#
# Polling strategy:
#   1. wait for k-matsu's PID 702377 to disappear
#   2. then poll nvidia-smi until every GPU has <2000 MiB used by anyone else
#   3. grace 30s, kick off the 5 scripts sequentially
#
# Output:
#   /tmp/base384_cfg07_sweep_wrapper.log   wrapper-level progress log
#   /tmp/eval_fid_base384_cfg07_*.log      per-script stdout (5 files)
#   results/halton_base384_cfg07_partialupdate_sweep.txt   parsed FID/IS rows
#   /tmp/base384_cfg07_sweep_DONE          marker created when all 5 finish
set -uo pipefail

cd /work/q-li/Halton-MaskGIT

WRAPPER_LOG=/tmp/base384_cfg07_sweep_wrapper.log
exec > >(tee -a "$WRAPPER_LOG") 2>&1

echo "============================================================"
echo "[$(date '+%F %T')] base-384 cfg=0.7 sweep wrapper started"
echo "============================================================"

# --- Step 1: wait for k-matsu's job (PID 702377) to finish -------------
TARGET_PID=702377
if kill -0 "$TARGET_PID" 2>/dev/null; then
  echo "[$(date '+%F %T')] waiting for PID $TARGET_PID (k-matsu/libero) to exit..."
  while kill -0 "$TARGET_PID" 2>/dev/null; do
    sleep 60
  done
  echo "[$(date '+%F %T')] PID $TARGET_PID exited."
else
  echo "[$(date '+%F %T')] PID $TARGET_PID not present; skipping wait."
fi

# --- Step 2: ensure all 4 GPUs are basically free -----------------------
echo "[$(date '+%F %T')] polling nvidia-smi for free GPUs (<2000 MiB)..."
while true; do
  MAX_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
              | tr -d ' ' | sort -n | tail -1)
  if [ -z "$MAX_USED" ]; then
    echo "[$(date '+%F %T')] nvidia-smi parse failed, retrying..."
    sleep 30
    continue
  fi
  if [ "$MAX_USED" -lt 2000 ]; then
    echo "[$(date '+%F %T')] all GPUs free (max used = ${MAX_USED} MiB)."
    break
  fi
  echo "[$(date '+%F %T')] GPUs still busy (max ${MAX_USED} MiB), wait 60s..."
  sleep 60
done

# --- Step 3: grace, then run 5 configs sequentially ---------------------
sleep 30
echo "[$(date '+%F %T')] launching 5-config sweep..."

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

run_one launch/eval_fid_base384_cfg07_partialupdate_N1.sh
run_one launch/eval_fid_base384_cfg07_partialupdate_N2.sh
run_one launch/eval_fid_base384_cfg07_partialupdate_N3.sh
run_one launch/eval_fid_base384_cfg07_partialupdate_N4.sh
run_one launch/eval_fid_base384_cfg07_partialupdate_norefresh.sh

# --- Step 4: marker -----------------------------------------------------
touch /tmp/base384_cfg07_sweep_DONE
echo "============================================================"
echo "[$(date '+%F %T')] base-384 cfg=0.7 sweep wrapper DONE"
echo "  results -> results/halton_base384_cfg07_partialupdate_sweep.txt"
echo "============================================================"
