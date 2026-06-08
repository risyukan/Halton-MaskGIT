#!/usr/bin/bash
# partial_update=True + 每 2 个 gated step 全量刷新一次缓存 (13/26 refresh)
# 跑完自动 append 一行到 results/halton_large384_partialupdate_sweep.txt
set -euo pipefail

cd /work/q-li/Halton-MaskGIT || exit 1

export HALTON_PARTIAL_UPDATE=1
export HALTON_CACHE_REFRESH_N=2

LOG=/tmp/eval_fid_partialupdate_N2.log
echo "[$(date '+%F %T')] start N=2 -> $LOG"

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

# 解析结果并 append 到 sweep 文件
FID=$(grep -oE "'FID': [0-9.]+" "$LOG" | tail -1 | grep -oE "[0-9.]+$" || echo "NA")
IS=$(grep -oE "'IS': [0-9.]+"   "$LOG" | tail -1 | grep -oE "[0-9.]+$" || echo "NA")
RT=$(grep -oE "[0-9]+:[0-9]+:[0-9]+<00:00"  "$LOG" | tail -1 | cut -d'<' -f1 || echo "NA")

printf "%-42s %-8s %-8s %-9s %s\n" \
  "partial_refresh_every_N2" "$FID" "$IS" "$RT" "13/26 refresh (50%)" \
  >> results/halton_large384_partialupdate_sweep.txt

echo "[$(date '+%F %T')] done  N=2  FID=$FID  IS=$IS  rt=$RT"
