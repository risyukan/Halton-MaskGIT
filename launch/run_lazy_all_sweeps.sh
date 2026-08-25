#!/usr/bin/bash
# 一次性把三个尺寸缺的 lazy r=1 refresh 实验全部跑完 (50k FID / IS)。
#
#   1) large-384 cfg=0.5  N=3            —— 补齐既有 sweep 缺的那一档
#   2) base-384  cfg=0.7  N=none,2,3,4,8,13
#   3) small-384 cfg=1.0  N=none,2,3,4,8,13
#
# 全部走 LAZY_RATIO=1.0 / lazy 覆盖全部层 / lazy head / seed=42 / step=32 /
# global_bsize=32 / fp32 / 4 卡 DDP, 与 large 既有 sweep 逐项对齐。
# 理论 FLOPs 与实测 latency 另行给出:
#   python flops_lazy_refresh.py {large,base,small}
#   python bench_latency_refresh.py 16 1 5 {large,base,small}   # 需独占单卡
#
# 开跑前先待机, 直到 4 张卡全空 (launch/wait_for_free_gpus.sh)。
# 只在最开始 gate 一次 —— 中途别人再占卡也不会打断已经排好的队列。
#
# 用法:
#   setsid nohup bash launch/run_lazy_all_sweeps.sh > /tmp/lazy_all_sweeps.log 2>&1 &
#   SKIP_WAIT=1 bash launch/run_lazy_all_sweeps.sh      # 不待机, 立刻开跑
set -uo pipefail

export PATH="/artic/q-li/miniconda3/envs/maskgit/bin:$PATH"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PROGRESS="$REPO/results/.lazy_all_sweeps_progress.txt"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$PROGRESS"; }

: > "$PROGRESS"
log "队列: large N=3 -> base N=0,2,3,4,8,13 -> small N=0,2,3,4,8,13 (共 13 个 50k eval)"

if [ "${SKIP_WAIT:-0}" != "1" ]; then
    bash launch/wait_for_free_gpus.sh 2>&1 | tee -a "$PROGRESS"
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        log "待机失败, 退出"
        exit 1
    fi
fi

run_one() {   # run_one <script> <N> <label>
    local script="$1" n="$2" label="$3"
    log "===== $label start ====="
    local start; start=$(date +%s)
    LAZY_RATIO=1.0 REFRESH_N="$n" bash "$script" 2>&1 | tee -a "$PROGRESS"
    local rc=${PIPESTATUS[0]}
    local el=$(( $(date +%s) - start ))
    log "===== $label done rc=$rc elapsed=$((el/3600))h$(( (el%3600)/60 ))m ====="
}

# 1) large 补 N=3
run_one launch/eval_fid_large384_lazycache.sh 3 "large N=3"

# 2) base / 3) small 全套
for SIZE in base small; do
    for N in ${NS:-0 2 3 4 8 13}; do
        run_one "launch/eval_fid_${SIZE}384_lazycache.sh" "$N" "$SIZE N=$N"
    done
done

log "ALL DONE"
