#!/usr/bin/bash
# Large-384 / cfg=0.5 — attn-delta cache (HALTON_ATTN_CACHE=1) + 现有 FFN cache,
#   NO cache refresh (pure partial, gated steps 5..30 全程走缓存)。
# 与 eval_fid_large384_partialupdate.sh 唯一差异 = 多开 HALTON_ATTN_CACHE=1。
# 对照基线 (FFN-only, 同设定) 见 results/halton_large384_partialupdate_sweep.txt:
#   partial_norefresh 6.0475
# 跑完自动 append 一行到 results/halton_large384_attncache_sweep.txt
set -euo pipefail

cd /work/q-li/Halton-MaskGIT || exit 1

export HALTON_PARTIAL_UPDATE=1
export HALTON_ATTN_CACHE=1          # <<< 本实验新增: attention inactive 用上一步缓存
export HALTON_CACHE_REFRESH_N=0     # 0 => 不周期刷新 (pure partial)

LOG=/tmp/eval_fid_large384_attncache_norefresh.log
echo "[$(date '+%F %T')] start large384 attn-cache no-refresh -> $LOG"

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
RT=$(grep -oE "[0-9]+:[0-9]+:[0-9]+<00:00"  "$LOG" | tail -1 | cut -d'<' -f1 || echo "NA")

printf "%-42s %-8s %-8s %-9s %s\n" \
  "attncache_no_refresh" "$FID" "$IS" "$RT" "0/26 refresh (pure partial)" \
  >> results/halton_large384_attncache_sweep.txt

echo "[$(date '+%F %T')] done  attn-cache no-refresh  FID=$FID  IS=$IS  rt=$RT"
