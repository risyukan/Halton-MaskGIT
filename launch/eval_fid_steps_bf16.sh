#!/usr/bin/bash
# step-reduction 对照实验 —— 单个配置。
#
#   bash launch/eval_fid_steps_bf16.sh <vit_size> <step> <config>
#     vit_size : large | base | small
#     step     : 采样步数 (32 | 24 | 20 | 16 | ...)
#     config   : baseline | N1..N6 | norefresh
#
# 目的: 回答"为什么不直接减少采样步数?"这个审稿人必问的问题。
# 需要两组曲线才能回答:
#   (1) baseline × 各 step  —— 不用任何 cache, 单纯减步数的 FID-延迟曲线
#   (2) layer cache × 各 step —— 证明本方法与减步数正交、可叠加
#
# 注意: step gate 会随 step 数按比例缩放 (Sampler/halton_sampler.py),
#       step=32 时为 5..30 与既有结果完全一致, 减步数时按同一进行率截取,
#       否则固定的"跳过前 5 步"在 step=16 会吃掉 31% 的步数, 无法横向比较。
set -euo pipefail
cd /work/q-li/Halton-MaskGIT || exit 1

VIT=${1:?vit_size required}
STEP=${2:?step required}
CFG=${3:?config required}

case "$VIT" in
  large) CFG_W=0.5; END_LAYER=23 ;;
  base)  CFG_W=0.7; END_LAYER=11 ;;
  small) CFG_W=1.0; END_LAYER=11 ;;
  *) echo "unknown vit_size: $VIT" >&2; exit 1 ;;
esac

export HALTON_ATTN_CACHE=0
unset HALTON_LAYER_CACHE_CLONE 2>/dev/null || true
unset HALTON_STEP_GATE_START   2>/dev/null || true   # step から自動導出させる
case "$CFG" in
  baseline)
    export HALTON_PARTIAL_UPDATE=0 HALTON_LAYER_CACHE=0 HALTON_CACHE_REFRESH_N=0
    NOTE="no cache" ;;
  norefresh)
    export HALTON_PARTIAL_UPDATE=1 HALTON_LAYER_CACHE=1 HALTON_CACHE_REFRESH_N=0
    NOTE="layer cache, no refresh" ;;
  N[1-9])
    export HALTON_PARTIAL_UPDATE=1 HALTON_LAYER_CACHE=1
    export HALTON_CACHE_REFRESH_N="${CFG#N}"
    NOTE="layer cache, refresh every ${CFG#N}" ;;
  *) echo "unknown config: $CFG" >&2; exit 1 ;;
esac
export HALTON_PARTIAL_START_LAYER=0
export HALTON_PARTIAL_END_LAYER=$END_LAYER

SWEEP="results/halton_${VIT}384_steps_bf16_sweep.txt"
LOG="/tmp/eval_fid_${VIT}384_steps_bf16_s${STEP}_${CFG}.log"

echo "[$(date '+%F %T')] start ${VIT}/step${STEP}/${CFG} -> $LOG"

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
  --sampler halton --step "$STEP" --cfg-w "$CFG_W" \
  --sm-temp 1.0 --sm-temp-min 1.0 --temp-warmup 1 \
  --sched-pow 2 --top-k -1 \
  --seed 42 > "$LOG" 2>&1

FID=$(grep -oE "'FID': [0-9.]+" "$LOG" | tail -1 | grep -oE "[0-9.]+$" || echo "NA")
IS=$(grep -oE "'IS': [0-9.]+"   "$LOG" | tail -1 | grep -oE "[0-9.]+$" || echo "NA")
# tqdm は 1 時間未満だと MM:SS になるので桁数を決め打ちしない
RT=$(tr '\r' '\n' < "$LOG" | grep -oE "\[(([0-9]+:)+[0-9]+)<00:00" | tail -1 \
     | sed 's/^\[//;s/<00:00$//' || echo "NA")

printf "%-8s %-11s %-9s %-10s %-9s %s\n" \
  "step${STEP}" "$CFG" "$FID" "$IS" "${RT:-NA}" "$NOTE" >> "$SWEEP"
echo "[$(date '+%F %T')] done  ${VIT}/step${STEP}/${CFG}  FID=$FID  IS=$IS  rt=${RT:-NA}"
