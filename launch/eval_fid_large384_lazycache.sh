#!/usr/bin/bash
# Large-384 / cfg=0.5 — LazyMAR Token Cache (HALTON_LAZY_CACHE=1)
#
# 与既有三个方案 (ffn delta / attn delta / layer output cache) 的唯一区别是
# "哪些 token 被重算"的决定方式:
#   既有方案 : active = U_{t-1} ∪ U_t         —— 纯 Halton 调度决定, 约 4~6% token
#   本方案   : active = U_{t-1} ∪ U_t ∪ register ∪ TopK(入口层 V 的余弦变化量)
#              名额由 LAZY_RATIO 给定 —— 内容自适应, 比例可调
#   LAZY_RATIO=1 (现默认) : 名额刚好被强制集合占满 → 完全不算 V 的余弦, 也就不
#              需要靠底层几层去攒打分材料 → 层区间默认放开到全部层。
# 执行机制沿用 LazyMAR: Q 只算 active token, K/V 走全长 per-layer 缓存
# (双向 self-attention 的上下文因此保持完整), FFN 只算 active token。
#
# 采样调度 (Halton 顺序、step gate 5..30、CFG、温度) 与 baseline 完全一致。
#
# 用法:
#   LAZY_RATIO=0.9 REFRESH_N=2 bash launch/eval_fid_large384_lazycache.sh
# 参数:
#   LAZY_RATIO  被缓存复用的 token 比例 r ∈ [0,1]; k = ceil((1-r)*N) 个 token 重算
#               r=1 (默认) → 只重算 U_t ∪ U_{t-1} ∪ register, 不打分
#               r=0 → 全部重算 → 逐位等于 baseline (正确性自检用)
#   LAZY_START / LAZY_END  lazy 覆盖的层区间, 默认 0..23 (全部层)
#   REFRESH_N   每 N 个 gated step 全量刷新一次缓存 (0 = 不刷新)
#               LazyMAR 原设定相当于 REFRESH_N=9
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

LAZY_RATIO="${LAZY_RATIO:-1.0}"
REFRESH_N="${REFRESH_N:-0}"
LAZY_START="${LAZY_START:-0}"                    # 默认全部 24 层都走 lazy
LAZY_END="${LAZY_END:-23}"

export HALTON_PARTIAL_UPDATE=1
export HALTON_LAZY_CACHE=1                       # <<< LazyMAR Token Cache
export HALTON_LAZY_CACHE_RATIO="$LAZY_RATIO"
export HALTON_LAZY_START_LAYER="$LAZY_START"
export HALTON_LAZY_END_LAYER="$LAZY_END"
export HALTON_CACHE_REFRESH_N="$REFRESH_N"
# 旧 sweep 的 layer gate (3..21) 用 LAZY_START=3 LAZY_END=21 复现
unset HALTON_PARTIAL_START_LAYER HALTON_PARTIAL_END_LAYER 2>/dev/null || true
# 与既有方案互斥, 显式关掉以免 transformer 直接报错
unset HALTON_LAYER_CACHE HALTON_ATTN_CACHE 2>/dev/null || true

TAG="lazycache_r${LAZY_RATIO}_N${REFRESH_N}_L${LAZY_START}-${LAZY_END}"
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
  --sampler halton --step 32 --cfg-w 0.5 \
  --sm-temp 1.0 --sm-temp-min 1.0 --temp-warmup 1 \
  --sched-pow 2 --top-k -1 \
  --seed 42 > "$LOG" 2>&1

FID=$(grep -oE "'FID': [0-9.]+" "$LOG" | tail -1 | grep -oE "[0-9.]+$" || echo "NA")
IS=$(grep -oE "'IS': [0-9.]+"   "$LOG" | tail -1 | grep -oE "[0-9.]+$" || echo "NA")
# tqdm 不到 1 小时只打 MM:SS, 超过才打 H:MM:SS —— 两种都要认, 否则 base/small
# 这种 <1h 的 eval 会记成 NA。
RT=$(grep -oE "[0-9]+:[0-9]+(:[0-9]+)?<00:00"  "$LOG" | tail -1 | cut -d'<' -f1 || echo "NA")

mkdir -p results
printf "%-42s %-8s %-8s %-9s %s\n" \
  "$TAG" "$FID" "$IS" "$RT" \
  "ratio=${LAZY_RATIO} refresh_every_${REFRESH_N} layers=${LAZY_START}..${LAZY_END}" \
  >> results/halton_large384_lazycache_sweep.txt

echo "[$(date '+%F %T')] done  ${TAG}  FID=$FID  IS=$IS  rt=$RT"
