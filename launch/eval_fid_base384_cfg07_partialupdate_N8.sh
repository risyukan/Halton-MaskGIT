#!/usr/bin/bash
# Base-384 / cfg=0.7 sweep — partial_update=True, refresh every 8 gated steps (4/26 refresh (15%))
# 跑完自动 append 一行到 results/halton_base384_cfg07_partialupdate_sweep.txt
set -euo pipefail

cd /work/q-li/Halton-MaskGIT || exit 1

export HALTON_PARTIAL_UPDATE=1
export HALTON_CACHE_REFRESH_N=8
# base 模型 depth=12; 保留末尾 2 层为 full update (i=10,11), 对齐 large 版的
# "末尾 2 层 full" 结构 => partial 作用层 = 3..9 (共 7 层)
export HALTON_PARTIAL_END_LAYER=9

LOG=/tmp/eval_fid_base384_cfg07_partialupdate_N8.log
echo "[$(date '+%F %T')] start base384 cfg=0.7 N=8 -> $LOG"

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  main.py \
  --mode cls-to-img --test-only --resume \
  --data-folder "" \
  --eval-folder "/work/Shared/Datasets/ILSVRC/ILSVRC2012/" \
  --vit-folder "./saved_networks/ImageNet_384_base.pth" \
  --vqgan-folder "./saved_networks/vq_ds16_c2i.pt" \
  --writer-log "./logs/" \
  --data imagenet --dtype float32 \
  --vit-size base --img-size 384 \
  --f-factor 16 --codebook-size 16384 --mask-value 16384 \
  --register 1 --proj 1 --dropout 0.1 \
  --nb-class 1000 --num-workers 8 --global-bsize 32 \
  --sampler halton --step 32 --cfg-w 0.7 \
  --sm-temp 1.0 --sm-temp-min 1.0 --temp-warmup 1 \
  --sched-pow 2 --top-k -1 \
  --seed 42 > "$LOG" 2>&1

FID=$(grep -oE "'FID': [0-9.]+" "$LOG" | tail -1 | grep -oE "[0-9.]+$" || echo "NA")
IS=$(grep -oE "'IS': [0-9.]+"   "$LOG" | tail -1 | grep -oE "[0-9.]+$" || echo "NA")
RT=$(grep -oE "[0-9]+:[0-9]+:[0-9]+<00:00"  "$LOG" | tail -1 | cut -d'<' -f1 || echo "NA")

printf "%-42s %-8s %-8s %-9s %s\n" \
  "partial_refresh_every_N8" "$FID" "$IS" "$RT" "4/26 refresh (15%)" \
  >> results/halton_base384_cfg07_partialupdate_sweep.txt

echo "[$(date '+%F %T')] done  N=8  FID=$FID  IS=$IS  rt=$RT"
