"""Halton-MaskGIT 的理论 FLOPs 分解图（数据由架构解析式计算，可复现）。

模型 (Config/base_cls2img.yaml, vit_size=large, proj=1):
  - self-attention only（无 cross attention）
  - SwiGLU FFN
  - d=1024, depth=24, heads=16, mlp_dim=4*d=4096
  - img 384 / f16 -> 24x24=576 token + 1 register = seq 577
  - head: Linear(d, codebook+1=16385)

两张图（海报用, 与旧版风格一致）:
  (a) Module-wise FLOPs   —— 彰显 transformer 占绝大部分
  (b) Main Transformer Breakdown —— 彰显 FFN 是最大计算负担 (仅 ffn / self_attn)

约定: FLOPs = 2 x MACs（乘加各计一次），单次前向 (batch=1)。
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator

# ---- 架构参数（来自 Config/base_cls2img.yaml, large）----
D = 1024                       # hidden_dim
DEPTH = 24                     # blocks
MLP_DIM = 4 * D                # 4096
SEQ = 24 * 24 + 1              # 576 tokens + 1 register = 577
CODEBOOK = 16384               # head 输出 = codebook + 1
MULT = 256


def swiglu_hidden(mlp_dim, multiple_of=MULT):
    h = int(2 * mlp_dim / 3)
    return multiple_of * ((h + multiple_of - 1) // multiple_of)


H = swiglu_hidden(MLP_DIM)     # 2816


def lin(n_in, n_out, tokens):
    """Linear 的 FLOPs (=2*MAC)。"""
    return 2 * tokens * n_in * n_out


# ---- 逐模块解析 FLOPs ----
# self-attention (每 block): wq/wk/wv/wo 4 个 d->d + QK^T + attn@V
attn_proj = 4 * lin(D, D, SEQ)
attn_mm = 2 * (2 * SEQ * SEQ * D)          # scores + context, 各 seq^2*d MAC
self_attn = (attn_proj + attn_mm) * DEPTH

# SwiGLU FFN (每 block): w1(d->h), w3(d->h), w2(h->d)
ffn = (2 * lin(D, H, SEQ) + lin(H, D, SEQ)) * DEPTH

transformer_layers = self_attn + ffn

# head: Linear(d, codebook+1)
head = lin(D, CODEBOOK + 1, SEQ)

# adaLN 调制 MLP (每 block Linear(d,6d), per-sample) + last_norm Linear(d,2d)
adaln = (DEPTH * lin(D, 6 * D, 1) + lin(D, 2 * D, 1))

# embed: cls/tok/pos/reg 全是查表, 计 ~0
embed = 1e7

MODULE = [
    ("transformer_layers", transformer_layers),
    ("head",               head),
    ("adaLN + norm",       adaln),
    ("embed",              embed),
]
BREAKDOWN = [
    ("ffn (SwiGLU)", ffn),
    ("self_attn",    self_attn),
]

# 强对比配色
C_HILITE = "#D1495B"   # 关键条
C_BASE = "#2E6F95"     # 其余条
BREAK_COLORS = ["#E8743B", "#2E6F95"]   # ffn 橙(突出) / self_attn 蓝

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 18,
    "axes.linewidth": 1.4,
    "font.family": "sans-serif",
})


def fmt_g(x, _=None):
    if x >= 1e12:
        return f"{x/1e12:.1f}T"
    if x >= 1e9:
        v = x / 1e9
        return (f"{v:.0f}G" if v >= 1 else f"{v:.1f}G")
    return f"{x/1e6:.0f}M"


def hbar(ax, items, title, colors, log=False):
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    y = range(len(labels))

    xmin = min(vals) / 3 if log else 0
    bars = ax.barh(y, vals, left=(xmin if log else 0),
                   color=colors, edgecolor="white", linewidth=1.2, zorder=3)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=18)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=24, fontweight="bold", pad=14)
    ax.set_xlabel("FLOPs (per forward pass)", fontsize=18)

    if log:
        ax.set_xscale("log")
        ax.set_xlim(xmin, max(vals) * 2.2)
        ax.xaxis.set_major_locator(LogLocator(base=10))
    else:
        ax.set_xlim(0, max(vals) * 1.32)
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_g))
    ax.tick_params(axis="x", labelsize=15)
    ax.grid(axis="x", linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)

    total = sum(vals)
    for b, v in zip(bars, vals):
        pct = 100 * v / total
        xr = b.get_width() + (b.get_x() if log else 0)
        off = xr * 0.08 if log else max(vals) * 0.015
        ax.text(xr + off, b.get_y() + b.get_height() / 2,
                f"{fmt_g(v)} ({pct:.1f}%)", va="center", ha="left",
                fontsize=15, fontweight="bold", color="#222")


def one_fig(items, title, colors, out_stem, figsize, log=False):
    fig, ax = plt.subplots(figsize=figsize)
    hbar(ax, items, title, colors, log=log)
    fig.text(0.99, 0.01, "Halton-MaskGIT large  d=1024  L=24  seq=577", ha="right",
             fontsize=12, color="#888")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = f"{out_stem}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"saved -> {out}")


def main():
    print(f"SwiGLU hidden = {H}")
    for name, v in MODULE:
        print(f"  {name:20s} {v/1e9:8.2f} G")
    print("  --")
    for name, v in BREAKDOWN:
        print(f"  {name:20s} {v/1e9:8.2f} G  ({100*v/transformer_layers:.1f}% of transformer)")

    mod_colors = [C_HILITE if k == "transformer_layers" else C_BASE
                  for k, _ in MODULE]
    one_fig(MODULE, "Module-wise FLOPs", mod_colors,
            "statics/flops_module_wise", (9.0, 4.2), log=True)

    one_fig(BREAKDOWN, "Main Transformer Breakdown", BREAK_COLORS,
            "statics/flops_transformer_breakdown", (9.0, 3.2), log=False)


if __name__ == "__main__":
    main()
