#!/usr/bin/bash
# layer-output cache (HALTON_LAYER_CACHE=1) 的 bf16 FID 评测 —— 单个配置。
#
# 用法:  bash launch/eval_fid_layercache_bf16.sh <vit_size> <config>
#   vit_size : large | base | small
#   config   : baseline | N1..N6 | norefresh
#
# 说明:
#   baseline  = partial_update 关闭 (全 token 更新), 作为 FID / latency 的分母
#   N<k>      = 每 k 个 gated step 全量刷新一次缓存 (N1 每步都刷 == baseline 计算量)
#   norefresh = 完全不刷新 (纯 partial)
#
# layer gate 一律取全层 (0..depth-1), 与 results/halton_large384_layercache_
# alllayers_sweep.txt 的做法一致。cfg_w 沿用各模型既有 sweep 的取值:
#   large 0.5 / base 0.7 / small 1.0 —— 保持与已有 fp32 结果可比。
set -euo pipefail
cd /work/q-li/Halton-MaskGIT || exit 1

VIT=${1:?vit_size required: large|base|small}
CFG=${2:?config required: baseline|N1..N6|norefresh}

case "$VIT" in
  large) CFG_W=0.5; END_LAYER=23 ;;
  base)  CFG_W=0.7; END_LAYER=11 ;;
  small) CFG_W=1.0; END_LAYER=11 ;;
  *) echo "unknown vit_size: $VIT" >&2; exit 1 ;;
esac

# ── 全部 HALTON_* 显式设置 (不依赖调用方环境) ──
export HALTON_ATTN_CACHE=0            # attn-delta cache 关闭
unset  HALTON_LAYER_CACHE_CLONE 2>/dev/null || true   # 默认走原地更新
case "$CFG" in
  baseline)
    export HALTON_PARTIAL_UPDATE=0
    export HALTON_LAYER_CACHE=0
    export HALTON_CACHE_REFRESH_N=0
    NOTE="reference (partial_update=False)"
    ;;
  norefresh)
    export HALTON_PARTIAL_UPDATE=1
    export HALTON_LAYER_CACHE=1
    export HALTON_CACHE_REFRESH_N=0
    NOTE="0/26 refresh (pure partial)"
    ;;
  N[1-9])
    export HALTON_PARTIAL_UPDATE=1
    export HALTON_LAYER_CACHE=1
    export HALTON_CACHE_REFRESH_N="${CFG#N}"
    NOTE="every ${CFG#N} gated steps"
    ;;
  *) echo "unknown config: $CFG" >&2; exit 1 ;;
esac
export HALTON_PARTIAL_START_LAYER=0
export HALTON_PARTIAL_END_LAYER=$END_LAYER

SWEEP="results/halton_${VIT}384_layercache_bf16_sweep.txt"
LOG="/tmp/eval_fid_${VIT}384_layercache_bf16_${CFG}.log"

echo "[$(date '+%F %T')] start ${VIT}/${CFG} (cfg_w=$CFG_W, gate 0..$END_LAYER, bf16) -> $LOG"

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  main.py \
  --mode cls-to-img --test-only --resume \
  --data-folder "" \
  --eval-folder "/work/Shared/Datasets/ILSVRC/ILSVRC2012/" \
  --vit-folder "./saved_networks/ImageNet_384_${VIT}.pth" \
  --vqgan-folder "./saved_networks/vq_ds16_c2i.pt" \
  --writer-log "./logs/" \
  --data imagenet --dtype bfloat16 \
  --vit-size "$VIT" --img-size 384 \
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

printf "%-14s %-9s %-10s %-9s %s\n" "$CFG" "$FID" "$IS" "$RT" "$NOTE" >> "$SWEEP"
echo "[$(date '+%F %T')] done  ${VIT}/${CFG}  FID=$FID  IS=$IS  rt=$RT"
