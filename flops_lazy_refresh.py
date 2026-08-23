"""lazy r=1 / 24 层 / lazy head 在不同 refresh 周期 N 下的理论 FLOPs 与加速比。

FLOPs 模型与 flops_lazy_token_cache.py 完全一致 (FLOPs = 2 x MACs), 唯一新增的是
refresh 步的处理 —— 语义严格照抄 Sampler/halton_sampler.py:225-231:

    is_refresh = (cache_refresh_n >= 1 and gated_step_counter % cache_refresh_n == 0)
    gated_step_counter += 1          # <<< 先判断后自增

即 gated_step_counter 从 0 开始, refresh 落在 counter = 0, N, 2N, ...
N=2 -> 13/26 步刷新, N=4 -> 7/26, 与既有 layercache sweep 的 notes 一致。
refresh 步走 active_mask=None 的全量分支, 与 baseline 单步同价。

用法: python flops_lazy_refresh.py
"""
import math

DEPTH, D = 24, 1024
MLP_DIM  = 4 * D
GRID     = 384 // 16
TOK      = GRID * GRID                # 576
REGISTER = 1
SEQ      = TOK + REGISTER             # 577
CODEBOOK = 16384
STEPS    = 32
GATE_LO, GATE_HI = 5, 31              # partial 步: GATE_LO <= t < GATE_HI


def swiglu_hidden(mlp_dim, multiple_of=256):
    h = int(2 * mlp_dim / 3)
    return multiple_of * ((h + multiple_of - 1) // multiple_of)


H = swiglu_hidden(MLP_DIM)


def lin(n_in, n_out, tokens):
    return 2 * tokens * n_in * n_out


def block_flops(q_tok, kv_tok, ctx_len):
    attn = (lin(D, D, q_tok)                  # wq
            + lin(D, D, kv_tok) * 2           # wk, wv
            + lin(D, D, q_tok)                # wo
            + 2 * (2 * q_tok * ctx_len * D))  # QK^T + attn@V
    ffn = 2 * lin(D, H, q_tok) + lin(H, D, q_tok)
    return attn + ffn


HEAD_FULL  = lin(D, CODEBOOK + 1, TOK)
ADALN      = DEPTH * lin(D, 6 * D, 1) + lin(D, 2 * D, 1)
FULL_BLOCK = block_flops(SEQ, SEQ, SEQ)
FULL_STEP  = FULL_BLOCK * DEPTH + HEAD_FULL + ADALN


def halton_r(t):
    r = 1 - (math.acos(min(1.0, (t + 1) / STEPS)) / (math.pi * 0.5))
    return max(t + 1, int(r * TOK))


R = [halton_r(t) for t in range(STEPS)]


def active_count(t):
    """|U_t ∪ U_{t-1}| + register."""
    r_t  = R[t]
    r_p1 = R[t - 1] if t >= 1 else 0
    r_p2 = R[t - 2] if t >= 2 else 0
    return (r_t - r_p1) + (r_p1 - r_p2) + REGISTER


def run(refresh_n):
    """返回 (总 FLOPs, refresh 步数, partial 步数, partial 平均 k/N)。"""
    total = 0.0
    counter = 0
    n_refresh = n_partial = 0
    ratios = []
    for t in range(STEPS):
        if GATE_LO <= t < GATE_HI:
            is_refresh = refresh_n >= 1 and counter % refresh_n == 0
            counter += 1
            if is_refresh:
                total += FULL_STEP
                n_refresh += 1
                continue
            k = active_count(t)
            total += (block_flops(k, k, SEQ) * DEPTH
                      + lin(D, CODEBOOK + 1, k)      # lazy head
                      + ADALN)
            n_partial += 1
            ratios.append(k / SEQ)
        else:
            total += FULL_STEP
    return total, n_refresh, n_partial, (sum(ratios) / len(ratios) if ratios else 0.0)


BASE = FULL_STEP * STEPS

print("== 理论 FLOPs | lazy r=1 | large-384 | 24/24 层 | lazy head | 32 step ==")
print(f"   depth={DEPTH} d={D} swiglu_h={H} seq={SEQ} | gate = [{GATE_LO}, {GATE_HI})")
print(f"   baseline = {BASE/1e9:.1f} GFLOPs/image\n")
print(f"{'refresh N':>10s} {'refresh步':>9s} {'partial步':>9s} {'全量步合计':>11s} "
      f"{'平均k/N':>8s} {'GFLOPs':>9s} {'FLOPs比':>8s} {'理论加速':>9s}")
for n in [0, 2, 4, 8, 13]:
    tot, nr, npart, kr = run(n)
    n_full = STEPS - (GATE_HI - GATE_LO) + nr
    label = "none" if n == 0 else str(n)
    print(f"{label:>10s} {nr:>9d} {npart:>9d} {n_full:>11d} "
          f"{kr:>8.4f} {tot/1e9:>9.1f} {tot/BASE:>8.4f} {BASE/tot:>8.3f}x")
