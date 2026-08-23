"""LazyMAR Token Cache (r=1, 全层) 相对原始 Halton-MaskGIT 的理论 FLOPs 加速。

FLOPs 用解析式算 (与 plot_flops_breakdown.py 同一套约定: FLOPs = 2 x MACs),
active token 数用采样器里真正的 Halton 调度算 —— 两者都不依赖任何实测。

方案 (HALTON_LAZY_CACHE=1, HALTON_LAZY_CACHE_RATIO=1.0, 全部层):
  * partial 步 (采样器 gate: 5 <= t < 31) 只对 active token 做 Q / wo / FFN,
    K/V 也只为 active token 重算后 scatter 进全长缓存 —— 于是每层每个 token 的
    全部矩阵乘都按 k/N 缩放, attention 的 QK^T / attn@V 因为 K/V 仍是全长, 从
    N x N 变成 k x N (同样按 k/N 缩放)。
  * active 集合 = U_t ∪ U_{t-1} ∪ register, 不做 V 相似度打分 (r=1)。
  * head (last_norm + Linear d->16385) 同样只对 active 行算, 其余行取上一步算好
    的 logit —— 逐 token 运算 + inactive 的 hidden 本来就是缓存值, 所以这一项是
    等价变换而非近似 (HALTON_LAZY_HEAD=0 可关掉做对照)。
  * 其余步 (t < 5, t = 31) 全量重算, 顺带整体刷新缓存。
  * gather / scatter / cache 回填是纯访存, 不计 FLOPs。

用法:  python flops_lazy_token_cache.py [vit_size] [img_size] [steps]
默认    python flops_lazy_token_cache.py large 384 32
"""
import math
import sys

VIT = {          # Network/transformer.py 的 vit_size -> (depth, hidden, heads)
    "tiny":   (12, 384),
    "small":  (12, 768),
    "base":   (24, 768),
    "large":  (24, 1024),
}

SIZE  = sys.argv[1] if len(sys.argv) > 1 else "large"
IMG   = int(sys.argv[2]) if len(sys.argv) > 2 else 384
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 32

DEPTH, D = VIT[SIZE]
MLP_DIM  = 4 * D
GRID     = IMG // 16                 # f_factor = 16
TOK      = GRID * GRID               # 图像 token
REGISTER = 1
SEQ      = TOK + REGISTER
CODEBOOK = 16384
MULT     = 256

# lazy 覆盖的层区间 (默认全部层) 与采样器的 step gate
LAZY_START, LAZY_END = 0, DEPTH - 1
GATE_LO, GATE_HI = 5, 31             # partial 步: GATE_LO <= t < GATE_HI


def swiglu_hidden(mlp_dim, multiple_of=MULT):
    h = int(2 * mlp_dim / 3)
    return multiple_of * ((h + multiple_of - 1) // multiple_of)


H = swiglu_hidden(MLP_DIM)


def lin(n_in, n_out, tokens):
    return 2 * tokens * n_in * n_out


def block_flops(q_tok, kv_tok, ctx_len):
    """一个 Block 的 FLOPs。

    q_tok  : 走 Q / wo / FFN 的 token 数
    kv_tok : 本步重算 K/V 的 token 数
    ctx_len: attention 的 K/V 长度 (缓存让它始终是全长 SEQ)
    """
    attn = (lin(D, D, q_tok)                 # wq
            + lin(D, D, kv_tok) * 2          # wk, wv
            + lin(D, D, q_tok)               # wo
            + 2 * (2 * q_tok * ctx_len * D)) # QK^T + attn@V
    ffn = 2 * lin(D, H, q_tok) + lin(H, D, q_tok)
    return attn + ffn


HEAD  = lin(D, CODEBOOK + 1, TOK)            # last_norm 后的 Linear, 全长
ADALN = DEPTH * lin(D, 6 * D, 1) + lin(D, 2 * D, 1)   # per-sample, 与 token 数无关

FULL_BLOCK = block_flops(SEQ, SEQ, SEQ)
FULL_STEP  = FULL_BLOCK * DEPTH + HEAD + ADALN


# ---- 采样器的 Halton 调度: 每步释放的 token 数 ----------------------------
def halton_r(step_idx):
    """Sampler/halton_sampler.py 里的累计已释放 token 数 r_t。"""
    ratio = (step_idx + 1) / STEPS
    r = 1 - (math.acos(min(1.0, ratio)) / (math.pi * 0.5))
    r = int(r * TOK)
    return max(step_idx + 1, r)


R = [halton_r(t) for t in range(STEPS)]


def active_count(t):
    """|U_t ∪ U_{t-1}| + register, 与 transformer 收到的 forced_mask 一致。"""
    r_t   = R[t]
    r_p1  = R[t - 1] if t >= 1 else 0
    r_p2  = R[t - 2] if t >= 2 else 0
    return (r_t - r_p1) + (r_p1 - r_p2) + REGISTER


# ---- 逐步累加 -------------------------------------------------------------
n_lazy_layers = LAZY_END - LAZY_START + 1
base_total = lazy_total = 0.0
base_blocks = lazy_blocks = 0.0
lazy_head_total = 0.0
rows = []
for t in range(STEPS):
    partial = GATE_LO <= t < GATE_HI
    k = active_count(t) if partial else SEQ
    if partial:
        blk = (block_flops(k, k, SEQ) * n_lazy_layers
               + FULL_BLOCK * (DEPTH - n_lazy_layers))
        head = lin(D, CODEBOOK + 1, k - REGISTER)     # register 不进 head
    else:
        blk = FULL_BLOCK * DEPTH
        head = HEAD
    step_flops = blk + head + ADALN
    base_total += FULL_STEP
    lazy_total += step_flops
    base_blocks += FULL_BLOCK * DEPTH
    lazy_blocks += blk
    lazy_head_total += head
    rows.append((t, partial, k, k / SEQ, step_flops / FULL_STEP))

G = 1e9
print(f"\n== 理论 FLOPs | {SIZE}-{IMG} | depth={DEPTH} d={D} swiglu_h={H} | "
      f"seq={SEQ} ({TOK}+{REGISTER} reg) | {STEPS} step ==")
print(f"   lazy 层区间 = [{LAZY_START}, {LAZY_END}] (共 {n_lazy_layers}/{DEPTH} 层), "
      f"partial 步 = {GATE_LO} <= t < {GATE_HI}, r = 1.0 (无 V 打分)\n")

print(f"{'step':>4} {'mode':>8} {'active':>7} {'k/N':>7} {'FLOPs/step(full=1)':>19}")
for t, partial, k, frac, rel in rows:
    print(f"{t:4d} {'partial' if partial else 'full':>8} {k:7d} {frac:7.3f} {rel:19.3f}")

# 对照: 只做 token cache, head 仍每步全量 (HALTON_LAZY_HEAD=0)
lazy_fullhead = lazy_blocks + STEPS * HEAD + STEPS * ADALN

print(f"\nbaseline           : {base_total/G:10.1f} GFLOPs / image "
      f"(blocks {base_blocks/G:.1f}, head {STEPS*HEAD/G:.1f})")
print(f"lazy r=1 (head 全量): {lazy_fullhead/G:10.1f} GFLOPs / image "
      f"(blocks {lazy_blocks/G:.1f}, head {STEPS*HEAD/G:.1f})")
print(f"lazy r=1 (head lazy): {lazy_total/G:10.1f} GFLOPs / image "
      f"(blocks {lazy_blocks/G:.1f}, head {lazy_head_total/G:.1f})")
print(f"\nFLOPs 比例 / 理论加速")
print(f"  仅 transformer 层        : {lazy_blocks/base_blocks:.4f}  "
      f"→ {base_blocks/lazy_blocks:.3f}x")
print(f"  端到端, head 全量        : {lazy_fullhead/base_total:.4f}  "
      f"→ {base_total/lazy_fullhead:.3f}x")
print(f"  端到端, head 也只算 active: {lazy_total/base_total:.4f}  "
      f"→ {base_total/lazy_total:.3f}x   <<< 默认")
part = [r for r in rows if r[1]]
print(f"\npartial 步平均 k/N = {sum(r[3] for r in part)/len(part):.4f} "
      f"({len(part)} 步), 全量步 {STEPS-len(part)} 步")
print(f"注: head 只算 active 是等价变换 (逐 token 运算 + inactive 的 hidden 本来就是"
      f"缓存值),\n    它把端到端加速从 {base_total/lazy_fullhead:.3f}x 提到 "
      f"{base_total/lazy_total:.3f}x, 已逼近 blocks-only 上限 {base_blocks/lazy_blocks:.3f}x。")
