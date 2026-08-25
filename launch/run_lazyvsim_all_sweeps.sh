#!/usr/bin/bash
# 一次性把 lazyvsim (LazyMAR 选点复刻, L3-最后一层) 缺的 50k FID/IS 全部跑完。
#
#   1) large-384 cfg=0.5  N=none,3,4,8,13   —— N=2 已在 sweep 里, 不重跑
#   2) base-384  cfg=0.7  N=none,2,3,4,8,13
#   3) small-384 cfg=1.0  N=none,2,3,4,8,13
#
# 全部走 VSIM_SCHED=lazymar / LAZY_START=3 / LAZY_END=depth-1 / lazy head /
# seed=42 / step=32 / global_bsize=32 / fp32 / 4 卡 DDP, 与既有 lazycache 和
# large 的 lazyvsim N=2 逐项对齐 (同 sampler / 同 gate / 同 seed), 严格可比。
#
# 理论 FLOPs 与实测 latency 另行给出:
#   python flops_lazy_vsim.py {large,base,small}
#   python bench_latency_vsim_sweep.py 16 1 5 {large,base,small}   # 需独占单卡
#
# 开跑前先待机, 直到 4 张卡全空 (launch/wait_for_free_gpus.sh)。
# 只在最开始 gate 一次 —— 中途别人再占卡也不会打断已经排好的队列。
#
# 用法:
#   setsid nohup bash launch/run_lazyvsim_all_sweeps.sh > /tmp/lazyvsim_all_sweeps.log 2>&1 &
#   SKIP_WAIT=1 bash launch/run_lazyvsim_all_sweeps.sh      # 不待机, 立刻开跑
#   SIZES="base small" NS="0 2" bash launch/run_lazyvsim_all_sweeps.sh   # 只跑一部分
set -uo pipefail

export PATH="/artic/q-li/miniconda3/envs/maskgit/bin:$PATH"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PROGRESS="$REPO/results/.lazyvsim_all_sweeps_progress.txt"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$PROGRESS"; }

: > "$PROGRESS"
log "队列: large N=0,3,4,8,13 -> base N=0,2,3,4,8,13 -> small N=0,2,3,4,8,13 (共 17 个 50k eval)"

if [ "${SKIP_WAIT:-0}" != "1" ]; then
    bash launch/wait_for_free_gpus.sh 2>&1 | tee -a "$PROGRESS"
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        log "待机失败, 退出"
        exit 1
    fi
fi

run_one() {   # run_one <size> <N> <lazy_end>
    local size="$1" n="$2" lend="$3"
    local label="$size N=$n"
    log "===== $label start ====="
    local start; start=$(date +%s)
    REFRESH_N="$n" LAZY_START=3 LAZY_END="$lend" \
        bash "launch/eval_fid_${size}384_lazyvsim.sh" 2>&1 | tee -a "$PROGRESS"
    local rc=${PIPESTATUS[0]}
    local el=$(( $(date +%s) - start ))
    log "===== $label done rc=$rc elapsed=$((el/3600))h$(( (el%3600)/60 ))m ====="
}

for SIZE in ${SIZES:-large base small}; do
    case "$SIZE" in
        large) LEND=23; DEF_NS="0 3 4 8 13" ;;    # N=2 已有结果
        *)     LEND=11; DEF_NS="0 2 3 4 8 13" ;;
    esac
    for N in ${NS:-$DEF_NS}; do
        run_one "$SIZE" "$N" "$LEND"
    done
done

log "ALL DONE"
