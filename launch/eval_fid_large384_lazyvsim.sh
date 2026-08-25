#!/usr/bin/bash
# Large-384 / cfg=0.5 — LazyMAR 选点复刻 (HALTON_LAZY_VSIM=1)
#
# 与 launch/eval_fid_large384_lazycache.sh 的唯一区别是"哪些 token 被重算"的
# 决定方式 —— 执行机制 (Q 只算 active / K-V 走全长 per-layer 缓存 / FFN 只算
# active / 出口层缓存回填 / head 只算 active) 与采样调度完全共用:
#
#   lazycache (r=1) : active = U_t ∪ U_{t-1} ∪ register     —— 纯 Halton 调度决定
#   本方案 (vsim)   : active = TopK_{rho_t*N} (1 - cos(V_l(t), V_l(t-1)))
#                     **没有任何强制 token** —— U_t / U_{t-1} / register 一律参与
#                     竞争, 名额 rho_t 取自 LazyMAR 的 RETAIN_RATIO_SCHEDULE,
#                     随 step 衰减 (64 步表按生成进度重采样到 32 步)。
#                     对应 LazyMAR/models/basic.py 的 _prune_tokens, 但去掉了那里
#                     的 score[mask_to_pred]=0 强制保留。
#
# 采样调度 (Halton 顺序、step gate 5..30、REFRESH_N、CFG、温度) 与 lazycache /
# baseline 逐步一致 —— 方法起效的 step 因此完全相同。
#
# 用法:
#   REFRESH_N=2 LAZY_START=3 bash launch/eval_fid_large384_lazyvsim.sh
# 参数:
#   LAZY_START / LAZY_END  lazy 覆盖的层区间, 默认 3..23 (LazyMAR 在 decoder
#                          layer 3 打分; start 必须 >= 1, 下面几层每步全量给打分
#                          攒材料)
#   REFRESH_N   每 N 个 gated step 全量刷新一次缓存 (0 = 不刷新)
#   VSIM_SCHED  重算比例表: "lazymar" (默认) / 常数 "0.3" / 逐步 "1,1,0.5,..."
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

REFRESH_N="${REFRESH_N:-2}"
LAZY_START="${LAZY_START:-3}"                    # LazyMAR 的 decoder layer 3
LAZY_END="${LAZY_END:-23}"                       # 一直用到最后一层
VSIM_SCHED="${VSIM_SCHED:-lazymar}"
CFG_W="${CFG_W:-0.5}"

export HALTON_PARTIAL_UPDATE=1
export HALTON_LAZY_CACHE=1                       # 执行机制沿用 LazyMAR Token Cache
export HALTON_LAZY_VSIM=1                        # <<< 选点: 纯 V 相似度, 无强制集合
export HALTON_LAZY_VSIM_SCHED="$VSIM_SCHED"
export HALTON_LAZY_START_LAYER="$LAZY_START"
export HALTON_LAZY_END_LAYER="$LAZY_END"
export HALTON_CACHE_REFRESH_N="$REFRESH_N"
# HALTON_LAZY_CACHE_RATIO 在 vsim 模式下不参与预算, 显式清掉以免误读。
unset HALTON_LAZY_CACHE_RATIO 2>/dev/null || true
unset HALTON_PARTIAL_START_LAYER HALTON_PARTIAL_END_LAYER 2>/dev/null || true
unset HALTON_LAYER_CACHE HALTON_ATTN_CACHE 2>/dev/null || true

TAG="lazyvsim_${VSIM_SCHED}_N${REFRESH_N}_L${LAZY_START}-${LAZY_END}_cfg${CFG_W}"
LOG="/tmp/eval_fid_large384_${TAG}.log"
echo "[$(date '+%F %T')] start ${TAG} -> $LOG"

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  main.py \
  --mode cls-to-img --test-only --resume \
  --data-folder "" \
  --eval-folder "/work/Shared/Datasets/ILSVRC/ILSVRC2012/" \
  --vit-folder "./saved_networks/ImageNet_384_large.pth" \
  --vqgan-folder "./saved_networks/vq_ds16_c2i.pt" \
  --writer-log "./logs/" \
  --data imagenet --dtype float32 \
  --vit-size large --img-size 384 \
  --f-factor 16 --codebook-size 16384 --mask-value 16384 \
  --register 1 --proj 1 --dropout 0.1 \
  --nb-class 1000 --num-workers 8 --global-bsize 32 \
  --sampler halton --step 32 --cfg-w "$CFG_W" \
  --sm-temp 1.0 --sm-temp-min 1.0 --temp-warmup 1 \
  --sched-pow 2 --top-k -1 \
  --seed 42 > "$LOG" 2>&1

FID=$(grep -oE "'FID': [0-9.]+" "$LOG" | tail -1 | grep -oE "[0-9.]+$" || echo "NA")
IS=$(grep -oE "'IS': [0-9.]+"   "$LOG" | tail -1 | grep -oE "[0-9.]+$" || echo "NA")
RT=$(grep -oE "[0-9]+:[0-9]+:[0-9]+<00:00"  "$LOG" | tail -1 | cut -d'<' -f1 || echo "NA")

mkdir -p results
printf "%-46s %-8s %-9s %-9s %s\n" \
  "$TAG" "$FID" "$IS" "$RT" \
  "sched=${VSIM_SCHED} refresh_every_${REFRESH_N} layers=${LAZY_START}..${LAZY_END} cfg=${CFG_W} no-forced-token" \
  >> results/halton_large384_lazyvsim_sweep.txt

echo "[$(date '+%F %T')] done  ${TAG}  FID=$FID  IS=$IS  rt=$RT"
