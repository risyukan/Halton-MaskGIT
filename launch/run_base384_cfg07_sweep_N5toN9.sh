#!/usr/bin/bash
# Run base-384 cfg=0.7 partial_update sweep for N=5,6,7,8,9 sequentially.
# All params identical to the existing N1..N4 scripts except HALTON_CACHE_REFRESH_N.
#
# Output:
#   /tmp/base384_cfg07_sweep_N5toN9_wrapper.log   wrapper-level progress
#   /tmp/eval_fid_base384_cfg07_partialupdate_N{5..9}.log   per-script stdout
#   results/halton_base384_cfg07_partialupdate_sweep.txt   parsed FID/IS rows
#   /tmp/base384_cfg07_sweep_N5toN9_DONE          marker created when all finish
set -uo pipefail

cd /work/q-li/Halton-MaskGIT

WRAPPER_LOG=/tmp/base384_cfg07_sweep_N5toN9_wrapper.log
exec > >(tee -a "$WRAPPER_LOG") 2>&1

echo "============================================================"
echo "[$(date '+%F %T')] base-384 cfg=0.7 sweep N5..N9 wrapper started"
echo "============================================================"

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

run_one launch/eval_fid_base384_cfg07_partialupdate_N5.sh
run_one launch/eval_fid_base384_cfg07_partialupdate_N6.sh
run_one launch/eval_fid_base384_cfg07_partialupdate_N7.sh
run_one launch/eval_fid_base384_cfg07_partialupdate_N8.sh
run_one launch/eval_fid_base384_cfg07_partialupdate_N9.sh

touch /tmp/base384_cfg07_sweep_N5toN9_DONE
echo "============================================================"
echo "[$(date '+%F %T')] base-384 cfg=0.7 sweep N5..N9 wrapper DONE"
echo "  results -> results/halton_base384_cfg07_partialupdate_sweep.txt"
echo "============================================================"
