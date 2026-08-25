#!/usr/bin/bash
# Small-384 / cfg=1.0 — LazyMAR 选点复刻 (HALTON_LAZY_VSIM=1)
#
# 与 launch/eval_fid_large384_lazyvsim.sh 逐项对齐, 唯一差别 = 模型尺寸
# (large 1024/24层 -> small 512/12层) 与 cfg_w (0.5 -> 1.0)。
# sampler / step / seed / global_bsize / dtype / 温度 / sched 全部不变, 严格可比。
#
# 选点 (与 large 版完全相同):
#   active = TopK_{ceil(rho_t*N)} (1 - cos(V_l3(t), V_l3(t-1)))
#   **没有任何强制 token** —— U_t / U_{t-1} / register 一律参与竞争,
#   rho_t 取自 LazyMAR 的 RETAIN_RATIO_SCHEDULE (64 步表按生成进度重采样到 32 步)。
#
# lazy 层区间: large 版是 L3-23 = "从 LazyMAR 的打分层 3 一路用到最后一层";
# small 只有 12 层, 同一语义 = L3-11。层 0..2 每步全量, 给入口层打分攒材料。
#
# 用法:
#   REFRESH_N=2 bash launch/eval_fid_small384_lazyvsim.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

REFRESH_N="${REFRESH_N:-2}"
LAZY_START="${LAZY_START:-3}"                    # LazyMAR 的 decoder layer 3
LAZY_END="${LAZY_END:-11}"                       # small depth=12 -> 一直用到最后一层
VSIM_SCHED="${VSIM_SCHED:-lazymar}"
CFG_W="${CFG_W:-1.0}"

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
  --sampler halton --step 32 --cfg-w "$CFG_W" \
  --sm-temp 1.0 --sm-temp-min 1.0 --temp-warmup 1 \
  --sched-pow 2 --top-k -1 \
  --seed 42 > "$LOG" 2>&1

FID=$(grep -oE "'FID': [0-9.]+" "$LOG" | tail -1 | grep -oE "[0-9.]+$" || echo "NA")
IS=$(grep -oE "'IS': [0-9.]+"   "$LOG" | tail -1 | grep -oE "[0-9.]+$" || echo "NA")
# tqdm 不到 1 小时只打 MM:SS, 超过才打 H:MM:SS —— 两种都要认。
RT=$(grep -oE "[0-9]+:[0-9]+(:[0-9]+)?<00:00"  "$LOG" | tail -1 | cut -d'<' -f1 || echo "NA")

mkdir -p results
printf "%-46s %-8s %-9s %-9s %s\n" \
  "$TAG" "$FID" "$IS" "$RT" \
  "sched=${VSIM_SCHED} refresh_every_${REFRESH_N} layers=${LAZY_START}..${LAZY_END} cfg=${CFG_W} no-forced-token" \
  >> results/halton_small384_lazyvsim_sweep.txt

echo "[$(date '+%F %T')] done  ${TAG}  FID=$FID  IS=$IS  rt=$RT"
