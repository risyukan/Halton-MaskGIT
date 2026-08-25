#!/usr/bin/bash
# 待机, 直到本机 4 张 A6000 全部空闲为止 (供 4 卡 DDP 的 50k FID eval 独占)。
#
# "空闲" 的判据 (两条同时满足, 且要连续命中 STABLE 次才算数):
#   1) nvidia-smi 里没有别人的 compute 进程 (自己的进程不算, 便于串行接力)
#   2) 每张卡 memory.used < MEM_FREE_MB 且 utilization.gpu < UTIL_FREE
#      —— 覆盖 (1) 抓不到的图形上下文 (别人的 sim 会开 type G context)。
# 连续命中的要求是为了避开别人两个 job 之间的空档。
#
# 环境变量:
#   NGPU=4  POLL=60(秒)  STABLE=3(次)  MEM_FREE_MB=1500  UTIL_FREE=10
#   MAX_WAIT=0  (秒, 0 = 无限等)
# 退出码: 0 = 已空闲可以开跑; 1 = 超过 MAX_WAIT 仍未空闲。
#
# 单独用: bash launch/wait_for_free_gpus.sh && bash launch/run_lazy_all_sweeps.sh
set -uo pipefail

NGPU="${NGPU:-4}"
POLL="${POLL:-60}"
STABLE="${STABLE:-3}"
MEM_FREE_MB="${MEM_FREE_MB:-1500}"
UTIL_FREE="${UTIL_FREE:-10}"
MAX_WAIT="${MAX_WAIT:-0}"
ME="$(id -un)"

log() { echo "[$(date '+%F %T')] [gpu-wait] $*"; }

others_on_gpu() {
    # 打印别人的 compute 进程 (pid:user), 没有就什么都不打印
    local pids p owner
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
    for p in $pids; do
        [ -z "$p" ] && continue
        owner=$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')
        [ -z "$owner" ] && continue
        [ "$owner" = "$ME" ] && continue
        echo "$p:$owner"
    done
}

gpus_quiet() {
    # 每张卡都要 mem/util 低于阈值
    local line idx mem util busy=0
    while IFS=, read -r idx mem util; do
        idx=$(echo "$idx" | tr -d ' '); mem=$(echo "$mem" | tr -d ' '); util=$(echo "$util" | tr -d ' ')
        [ -z "$idx" ] && continue
        if [ "$mem" -ge "$MEM_FREE_MB" ] || [ "$util" -ge "$UTIL_FREE" ]; then
            busy=1
            echo "gpu$idx(${mem}MiB/${util}%)"
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
                        --format=csv,noheader,nounits 2>/dev/null | head -n "$NGPU")
    return $busy
}

log "开始待机: 需要 $NGPU 张卡连续 $STABLE 次 (每 ${POLL}s 一次) 满足 mem<${MEM_FREE_MB}MiB 且 util<${UTIL_FREE}%, 且无他人 compute 进程"

hits=0
t0=$(date +%s)
while true; do
    others=$(others_on_gpu)
    busy=$(gpus_quiet); quiet_rc=$?

    if [ -z "$others" ] && [ "$quiet_rc" -eq 0 ]; then
        hits=$((hits + 1))
        log "空闲 ($hits/$STABLE)"
        if [ "$hits" -ge "$STABLE" ]; then
            log "4 张卡已连续 $STABLE 次空闲 —— 开跑"
            exit 0
        fi
    else
        [ "$hits" -gt 0 ] && log "空闲计数清零"
        hits=0
        reason=""
        [ -n "$others" ] && reason="他人进程 [$(echo "$others" | tr '\n' ' ')]"
        [ -n "$busy" ]   && reason="$reason 忙碌卡 [$busy]"
        log "仍被占用:$reason"
    fi

    if [ "$MAX_WAIT" -gt 0 ] && [ $(( $(date +%s) - t0 )) -ge "$MAX_WAIT" ]; then
        log "超过 MAX_WAIT=${MAX_WAIT}s 仍未空闲, 放弃"
        exit 1
    fi
    sleep "$POLL"
done
