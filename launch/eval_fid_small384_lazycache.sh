#!/usr/bin/bash
# Small-384 / cfg=1.0 — LazyMAR Token Cache (HALTON_LAZY_CACHE=1)
#
# 与 launch/eval_fid_large384_lazycache.sh 逐项对齐, 唯一差别 = 模型尺寸
# (large 1024/24层 -> small 512/12层) 与 cfg_w (0.5 -> 1.0)。
# sampler / step / seed / global_bsize / dtype / 温度 / sched 全部不变, 严格可比。
#
# 方案 (照抄 large 版):
#   LAZY_RATIO=1  -> active = U_t ∪ U_{t-1} ∪ register, 不算 V 余弦打分
#   lazy 覆盖全部层 (0..11, large 版是 0..23) —— 层数不同, "全部层"这一语义相同
#   Q/FFN 只算 active token; K/V 走 per-layer 全长缓存; head 也走 lazy
#   REFRESH_N 个 gated step 全量刷新一次 (0 = 不刷新)
#
# 用法:
#   REFRESH_N=2 bash launch/eval_fid_small384_lazycache.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

LAZY_RATIO="${LAZY_RATIO:-1.0}"
REFRESH_N="${REFRESH_N:-0}"
LAZY_START="${LAZY_START:-0}"                    # small depth=12 -> 全部层 = 0..11
LAZY_END="${LAZY_END:-11}"

export HALTON_PARTIAL_UPDATE=1
export HALTON_LAZY_CACHE=1                       # <<< LazyMAR Token Cache
export HALTON_LAZY_CACHE_RATIO="$LAZY_RATIO"
export HALTON_LAZY_START_LAYER="$LAZY_START"
export HALTON_LAZY_END_LAYER="$LAZY_END"
export HALTON_CACHE_REFRESH_N="$REFRESH_N"
unset HALTON_PARTIAL_START_LAYER HALTON_PARTIAL_END_LAYER 2>/dev/null || true
unset HALTON_LAYER_CACHE HALTON_ATTN_CACHE 2>/dev/null || true

TAG="lazycache_r${LAZY_RATIO}_N${REFRESH_N}_L${LAZY_START}-${LAZY_END}"
LOG="/tmp/eval_fid_small384_${TAG}.log"
echo "[$(date '+%F %T')] start small384 ${TAG} -> $LOG"

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  main.py \
  --mode cls-to-img --test-only --resume \
  --data-folder "" \
  --eval-folder "/work/Shared/Datasets/ILSVRC/ILSVRC2012/" \
  --vit-folder "./saved_networks/ImageNet_384_small.pth" \
  --vqgan-folder "./saved_networks/vq_ds16_c2i.pt" \
  --writer-log "./logs/" \
  --data imagenet --dtype float32 \
  --vit-size small --img-size 384 \
  --f-factor 16 --codebook-size 16384 --mask-value 16384 \
  --register 1 --proj 1 --dropout 0.1 \
  --nb-class 1000 --num-workers 8 --global-bsize 32 \
  --sampler halton --step 32 --cfg-w 1.0 \
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
  >> results/halton_small384_lazycache_sweep.txt

echo "[$(date '+%F %T')] done  ${TAG}  FID=$FID  IS=$IS  rt=$RT"
