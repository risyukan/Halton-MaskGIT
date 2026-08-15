#!/usr/bin/bash
# Large-384 / cfg=0.5 — layer-output cache (HALTON_LAYER_CACHE=1), layer 全开
# (partial 门限 0..23, 即所有 24 层在 partial 步都走 layer cache); ffn/attn cache 关闭。
# 完全不刷新 (N=0 => pure partial, 0/26 refresh): 极限加速 / 质量下限。
# 跑完自动 append 一行到 results/halton_large384_layercache_alllayers_sweep.txt
set -euo pipefail

cd /work/q-li/Halton-MaskGIT || exit 1

export HALTON_PARTIAL_UPDATE=1
export HALTON_LAYER_CACHE=1          # <<< layer-output cache (关掉 ffn/attn cache 路径)
export HALTON_PARTIAL_START_LAYER=0  # layer 全开: 从第 0 层
export HALTON_PARTIAL_END_LAYER=23   #            到第 23 层 (large 共 24 层)
export HALTON_CACHE_REFRESH_N=0      # 0 => no periodic refresh (pure partial)
unset HALTON_ATTN_CACHE 2>/dev/null || true   # 确保 attention cache 关闭

LOG=/tmp/eval_fid_large384_layercache_alllayers_norefresh.log
echo "[$(date '+%F %T')] start layercache all-layers no-refresh -> $LOG"

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
  "layercache_alllayers_norefresh" "$FID" "$IS" "$RT" "0/26 refresh (pure partial)" \
  >> results/halton_large384_layercache_alllayers_sweep.txt

echo "[$(date '+%F %T')] done  layercache all-layers no-refresh  FID=$FID  IS=$IS  rt=$RT"
