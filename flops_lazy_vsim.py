"""LazyMAR 选点复刻 (HALTON_LAZY_VSIM) 在不同 refresh 周期 N 下的理论 FLOPs 与加速比。

FLOPs 模型与 flops_lazy_refresh.py / flops_lazy_token_cache.py 完全一致
(FLOPs = 2 x MACs, 只数线性层与两个 attention matmul; RMSNorm / SiLU / 余弦打分
这些逐元素算子不计), 与 lazycache 的差别全部来自 vsim 的两条结构性改动:

  1) lazy 区间是 [LAZY_START=3, depth-1] 而不是 [0, depth-1] ——
     第 0..2 层每步全量, 给入口层的 V 打分攒材料 (LazyMAR 在 decoder layer 3 打分)。
  2) partial 步的预算不是"调度定死的 U_t ∪ U_{t-1} ∪ register", 而是
     k_t = ceil(rho_t * N_tok), rho_t = LazyMAR 的 RETAIN_RATIO_SCHEDULE
     (64 步表按生成进度重采样到 32 步, 见 Network/transformer.py:_lazymar_ratio)。

单个 partial 步的成本 (逐项对应 TransformerEncoder._forward_lazy):

    rho_t = 1  ->  退化成全量步 (代码走 `elif vsim` 分支, 不打分也不省算力),
                   唯一差别是 head 算 577 行而不是 576 行。
    rho_t < 1  ->  层 0..start-1        : 全量 block
                   层 start (入口层)     : K/V 全长 (打分要用), Q/FFN 只算 k 行
                                          -> block_flops(k, SEQ, SEQ)
                   层 start+1..end      : K/V 也只算 k 行, attention 仍对全长
                                          -> block_flops(k, k, SEQ)
                   层 end+1..depth-1    : 全量 block (end = depth-1 时没有)
                   head                 : 只算 k 行 (HALTON_LAZY_HEAD=1)

refresh 语义严格照抄 Sampler/halton_sampler.py:225-231 (先判断后自增, counter 从
0 开始, 故 refresh 落在 counter = 0, N, 2N, ...), 与 lazycache 逐步一致 ——
两个方案在同一个 N 下起效的 step 完全相同, 可直接对比。

用法:
    python flops_lazy_vsim.py                 # large / base / small 都打印
    python flops_lazy_vsim.py base            # 只打印 base
    LAZY_START=1 python flops_lazy_vsim.py    # 换 lazy 区间起点
被 bench_latency_vsim_sweep.py import 时, 先 set_size(size) 再用 run()/BASE。
"""
import math
import os
import sys

# ── 与尺寸无关的常量 (与 flops_lazy_refresh.py 同源) ──────────────────────
GRID     = 384 // 16
TOK      = GRID * GRID                # 576
REGISTER = 1
SEQ      = TOK + REGISTER             # 577
CODEBOOK = 16384
STEPS    = 32
GATE_LO, GATE_HI = 5, 31              # partial 步: GATE_LO <= t < GATE_HI

SIZES = {                             # name -> (depth, hidden_dim)
    "tiny":  (6, 384),
    "small": (12, 512),
    "base":  (12, 768),
    "large": (24, 1024),
    "xlarge": (28, 1152),
}

REFRESH_NS = [0, 2, 3, 4, 8, 13]      # 0 = no refresh

# lazy 区间起点: LazyMAR 在 decoder layer 3 打分, 这里默认一致。
# 终点默认 depth-1 (一路用到最后一层) —— large 即 L3-23, base/small 即 L3-11。
LAZY_START = int(os.environ.get("LAZY_START", "3"))
LAZY_END   = os.environ.get("LAZY_END")   # None -> depth-1

# LazyMAR/models/basic.py 的 RETAIN_RATIO_SCHEDULE (逐位与 transformer.py 一致)
LAZYMAR_RETAIN_RATIO_SCHEDULE = [
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    0.6, 0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4,
    0.15, 0.15, 0.15, 0.15, 0.15, 0.12, 0.12, 0.12, 0.12, 0.12,
    0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
    0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
    0.05, 0.05, 0.05, 0.05,
]


def lazymar_ratio(step, total_steps=STEPS):
    n = len(LAZYMAR_RETAIN_RATIO_SCHEDULE)
    idx = int(int(step) * n / max(1, int(total_steps)))
    return LAZYMAR_RETAIN_RATIO_SCHEDULE[min(max(idx, 0), n - 1)]


def budget(step):
    """k_t = ceil(rho_t * N_tok), 与 _forward_lazy 里的 clamp 完全一致。"""
    rho = min(max(lazymar_ratio(step), 0.0), 1.0)
    return max(1, min(SEQ, int(math.ceil(rho * SEQ))))


def swiglu_hidden(mlp_dim, multiple_of=256):
    h = int(2 * mlp_dim / 3)
    return multiple_of * ((h + multiple_of - 1) // multiple_of)


def lin(n_in, n_out, tokens):
    return 2 * tokens * n_in * n_out


def block_flops(q_tok, kv_tok, ctx_len):
    attn = (lin(D, D, q_tok)                  # wq
            + lin(D, D, kv_tok) * 2           # wk, wv
            + lin(D, D, q_tok)                # wo
            + 2 * (2 * q_tok * ctx_len * D))  # QK^T + attn@V
    ffn = 2 * lin(D, H, q_tok) + lin(H, D, q_tok)
    return attn + ffn


def set_size(size, start=None, end=None):
    """把模块级 globals 切到指定尺寸。返回 (depth, dim)。"""
    global SIZE, DEPTH, D, MLP_DIM, H, HEAD_FULL, ADALN, FULL_BLOCK, FULL_STEP, BASE
    global L_START, L_END
    if size not in SIZES:
        raise ValueError(f"unknown vit size {size!r}, expect one of {list(SIZES)}")
    SIZE = size
    DEPTH, D = SIZES[size]
    MLP_DIM = 4 * D
    H = swiglu_hidden(MLP_DIM)
    HEAD_FULL = lin(D, CODEBOOK + 1, TOK)
    ADALN = DEPTH * lin(D, 6 * D, 1) + lin(D, 2 * D, 1)
    FULL_BLOCK = block_flops(SEQ, SEQ, SEQ)
    FULL_STEP = FULL_BLOCK * DEPTH + HEAD_FULL + ADALN
    BASE = FULL_STEP * STEPS
    _s = LAZY_START if start is None else int(start)
    _e = (DEPTH - 1) if end is None else int(end)
    if LAZY_END is not None and end is None:
        _e = int(LAZY_END)
    L_END = max(0, min(_e, DEPTH - 1))
    L_START = max(1, min(_s, L_END))        # vsim 要求 start >= 1
    return DEPTH, D


def partial_step_flops(k):
    """一个 partial 步的 FLOPs (k = 本步重算的 token 数)。"""
    if k >= SEQ:
        # rho_t = 1: 走 `elif vsim` 分支 —— 全部层全量, head 算 SEQ 行
        return FULL_BLOCK * DEPTH + lin(D, CODEBOOK + 1, SEQ) + ADALN
    total = FULL_BLOCK * L_START                     # 层 0..start-1: 全量
    total += block_flops(k, SEQ, SEQ)                # 入口层: 全长 K/V + active Q/FFN
    total += block_flops(k, k, SEQ) * (L_END - L_START)   # 区间内其余层
    total += FULL_BLOCK * (DEPTH - 1 - L_END)        # 区间以上: 全量
    total += lin(D, CODEBOOK + 1, k)                 # lazy head
    total += ADALN
    return total


def run(refresh_n):
    """返回 (总 FLOPs, refresh 步数, partial 步数, partial 平均 k/N, 退化成全量的 partial 步数)。"""
    total = 0.0
    counter = 0
    n_refresh = n_partial = n_degen = 0
    ratios = []
    for t in range(STEPS):
        if GATE_LO <= t < GATE_HI:
            is_refresh = refresh_n >= 1 and counter % refresh_n == 0
            counter += 1
            if is_refresh:
                total += FULL_STEP
                n_refresh += 1
                continue
            k = budget(t)
            total += partial_step_flops(k)
            n_partial += 1
            n_degen += (k >= SEQ)
            ratios.append(k / SEQ)
        else:
            total += FULL_STEP
    return (total, n_refresh, n_partial,
            (sum(ratios) / len(ratios) if ratios else 0.0), n_degen)


def report(size):
    set_size(size)
    print(f"== 理论 FLOPs | lazyvsim (LazyMAR 选点) | {size}-384 | "
          f"lazy 层 L{L_START}-{L_END}/{DEPTH-1} | lazy head | 32 step ==")
    print(f"   depth={DEPTH} d={D} swiglu_h={H} seq={SEQ} | gate = [{GATE_LO}, {GATE_HI})")
    print(f"   baseline = {BASE/1e9:.1f} GFLOPs/image\n")
    print(f"{'refresh N':>10s} {'refresh步':>9s} {'partial步':>9s} {'其中rho=1':>9s} "
          f"{'全量步合计':>11s} {'平均k/N':>8s} {'GFLOPs':>9s} {'FLOPs比':>8s} {'理论加速':>9s}")
    for n in REFRESH_NS:
        tot, nr, npart, kr, ndeg = run(n)
        n_full = STEPS - (GATE_HI - GATE_LO) + nr
        label = "none" if n == 0 else str(n)
        print(f"{label:>10s} {nr:>9d} {npart:>9d} {ndeg:>9d} {n_full:>11d} "
              f"{kr:>8.4f} {tot/1e9:>9.1f} {tot/BASE:>8.4f} {BASE/tot:>8.3f}x")
    print()
    print("   注: 'rho=1' 列 = LazyMAR 表前段取 1.0 而退化成全量的 partial 步 —— 这些步"
          "完全不省算力。")
    print()


set_size(os.environ.get("LAZY_VIT_SIZE", "large"))

if __name__ == "__main__":
    targets = sys.argv[1:] or ["large", "base", "small"]
    for s in targets:
        report(s)
