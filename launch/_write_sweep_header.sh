#!/usr/bin/bash
# sweep ファイルのヘッダを出力する (run_layercache_bf16_sweep.sh から呼ばれる)。
VIT=${1:?vit_size required}
case "$VIT" in
  large) CFG_W=0.5; DEPTH=24; DIM="hidden=1024/24L/16H, 480M" ;;
  base)  CFG_W=0.7; DEPTH=12; DIM="hidden=768/12L/12H, ~155M" ;;
  small) CFG_W=1.0; DEPTH=12; DIM="hidden=512/12L/6H,  ~78M" ;;
esac
END=$((DEPTH - 1))

cat <<EOF
# Halton-MaskGIT — ${VIT}-384 / layer-output cache / bfloat16 sweep
#
# Model      : ImageNet_384_${VIT}.pth (${DIM})
# Method     : HALTON_LAYER_CACHE=1 (每层缓存整层输出; partial 步只重算 active
#              token, Q 取 active / K/V 取全部, FFN 只算 active; inactive 位置沿用
#              缓存的整层输出, 缓存原地更新)。ffn/attn cache 均关闭。
# Sampler    : halton, step=32, cfg_w=${CFG_W}, sm_temp=1.0, sm_temp_min=1.0,
#              temp_warmup=1, sched_pow=2, top_k=-1
# dtype      : bfloat16
# GPUs       : 4 (DDP), global_bsize=32
# num_images : 50000
# seed       : 42
# ref stats  : saved_networks/ImageNet_256_train_stats.pt
# step gate  : partial_update active in halton steps 5..30 (前 5 步/最后 1 步全量)
# layer gate : 0..${END} (全 ${DEPTH} 层)
# eval cmd   : bash launch/run_layercache_bf16_sweep.sh ${VIT}
#
# config 说明:
#   baseline  = partial_update 关闭 (全 token 更新), FID / latency 的分母
#   N<k>      = 每 k 个 gated step 全量刷新一次 (N1 每个 gated step 都刷, 计算量 == baseline)
#   norefresh = 完全不刷新 (纯 partial)
#
# latency 见 results/latency_${VIT}384_layercache_bf16.csv
#   (单卡 A6000, batch 8, transformer-only, 与本表同 cfg_w / 同 layer gate)
#
# columns: config  FID  IS  runtime  notes
# ------------------------------------------------------------
EOF
