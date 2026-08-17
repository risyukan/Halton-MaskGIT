#!/usr/bin/bash
VIT=${1:?vit_size required}
case "$VIT" in
  large) CFG_W=0.5; DEPTH=24 ;;
  base)  CFG_W=0.7; DEPTH=12 ;;
  small) CFG_W=1.0; DEPTH=12 ;;
esac
END=$((DEPTH - 1))

cat <<EOF
# Halton-MaskGIT — ${VIT}-384 / step-reduction 对照 / bfloat16
#
# 目的: 回答"为什么不直接减少采样步数?" —— 加速类论文最先被问到的问题。
#   baseline 列 : 不用任何 cache, 单纯把 step 从 32 减到 16 的 FID-延迟曲线
#   N2 列       : 在各 step 之上叠加 layer cache (refresh N=2), 验证两者正交
# 两条曲线画在同一张 FID-vs-latency 图上, 才能判断本方法是否真的优于"少跑几步"。
#
# Model      : ImageNet_384_${VIT}.pth
# Sampler    : halton, cfg_w=${CFG_W}, sm_temp=1.0, sm_temp_min=1.0,
#              temp_warmup=1, sched_pow=2, top_k=-1
# dtype      : bfloat16 | GPUs: 4 (DDP), global_bsize=32 | 50000 imgs | seed 42
# layer gate : 0..${END} (全 ${DEPTH} 层)
# step gate  : gate_start = round(5 * step / 32), 到 step-2 为止
#              (step=32 时为 5..30, 与 halton_${VIT}384_layercache_bf16_sweep.txt 一致;
#               若固定为"跳过前 5 步", step=16 会吃掉 31% 的步数而无法横向比较)
# eval cmd   : bash launch/run_steps_bf16_sweep.sh ${VIT}
#
# columns: step  config  FID  IS  runtime  notes
# ------------------------------------------------------------
EOF
