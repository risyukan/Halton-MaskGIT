#!/usr/bin/bash
# lazy r=1 / 24 层 / lazy head —— refresh N ∈ {0(none), 2, 4, 8, 13} 的 50k FID 扫描。
#
# 每档都调用 launch/eval_fid_large384_lazycache.sh, 只改 REFRESH_N;
# LAZY_RATIO=1.0 / LAZY_START=0 / LAZY_END=23 / HALTON_LAZY_HEAD 默认(=1) 保持不变。
# 顺序跑, 每档独占 4 卡。结果由 eval 脚本自己 append 到
# results/halton_large384_lazycache_sweep.txt。
set -uo pipefail

export PATH="/artic/q-li/miniconda3/envs/maskgit/bin:$PATH"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PROGRESS="$REPO/results/.lazy_refresh_sweep_progress.txt"
: > "$PROGRESS"

for N in 0 2 4 8 13; do
  echo "[$(date '+%F %T')] ===== REFRESH_N=$N start =====" | tee -a "$PROGRESS"
  START=$(date +%s)
  LAZY_RATIO=1.0 REFRESH_N=$N bash launch/eval_fid_large384_lazycache.sh 2>&1 | tee -a "$PROGRESS"
  RC=${PIPESTATUS[0]}
  ELAPSED=$(( $(date +%s) - START ))
  echo "[$(date '+%F %T')] ===== REFRESH_N=$N done rc=$RC elapsed=$((ELAPSED/3600))h$(( (ELAPSED%3600)/60 ))m =====" | tee -a "$PROGRESS"
done

echo "[$(date '+%F %T')] ALL DONE" | tee -a "$PROGRESS"
