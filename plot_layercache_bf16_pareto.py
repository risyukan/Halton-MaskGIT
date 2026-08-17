"""layer cache (bf16) の FID–速度トレードオフ図。

データは手打ちせず、実験成果物からそのまま読む:
    results/halton_{large,base,small}384_layercache_bf16_sweep.txt   … FID / IS
    results/latency_{large,base,small}384_layercache_bf16.csv        … 加速比

左 (a): FID vs 端到端加速比 — baseline から norefresh までの全域。
        3 モデルの絶対 FID と、refresh を切ったときの崩れ方が見える。
右 (b): ΔFID vs 端到端加速比 — N=2..6 の実用域だけを拡大。
        「モデルが小さいほど曲線が平ら = キャッシュに強い」を見せる本題。

x 軸は端到端 (VQGAN decode + サンプラ開銷込み)。transformer-only の加速比は
これより高い (例 large N2: 1.335x vs 1.296x) が、報告値としては端到端を採る。

出力: statics/layercache_bf16_pareto.{pdf,png}
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── パレット (dataviz スキルの参照パレット light / 先頭 3 スロット) ──
# scripts/validate_palette.js "#2a78d6,#eb6834,#1baf7a" --mode light --pairs all
#   → ALL CHECKS PASS (最悪 CVD ΔE 9.2 / 通常視 24.0)
#   → WARN: aqua は対 surface コントラスト 2.74 (<3:1)。marker 形状・直接ラベル・
#           凡例という二次符号化を必ず併用すること (本図はいずれも満たす)。
MODELS = [
    # key,     label,               color,      marker, (a)端点ラベル位置, (b)ラベル位置
    ("large", "H-MaskGIT-L (480M)", "#2a78d6", "o", (-7, 4, "right"), (0, 9, "center")),
    ("base",  "H-MaskGIT-B (155M)", "#eb6834", "s", (7, -3, "left"), (0, 0, "center")),
    ("small", "H-MaskGIT-S (78M)",  "#1baf7a", "^", (-7, 4, "right"), (0, 9, "center")),
]
INK        = "#0b0b0b"    # text-primary
INK_SOFT   = "#52514e"    # text-secondary
INK_MUTED  = "#8a8880"    # 罫線・軸
SURFACE    = "#fcfcfb"

CONFIGS = ["N1", "N2", "N3", "N4", "N5", "N6", "norefresh"]
# 直接ラベルは要点だけに絞る (全点に数字を振らない)。
#   (a) は端点 (no refresh) のみ — この図の主張は「崩れ方」であって個々の N ではない
#   (b) は両端 N=2 / N=6 のみ、系列ごとに上下へ振り分けて衝突を避ける
LABELLED_A = {"norefresh"}
LABELLED_B = {"N2", "N6"}
LABEL_OFF_B = {
    ("large", "N2"): (0, 9, "center"),   ("large", "N6"): (0, 9, "center"),
    ("base",  "N2"): (9, -3, "left"),
    ("small", "N2"): (0, 9, "center"),
}
PRETTY   = {"N1": "N=1", "N2": "N=2", "N3": "N=3", "N4": "N=4",
            "N5": "N=5", "N6": "N=6", "norefresh": "no refresh"}


def read_fid(vit):
    """sweep txt から config -> FID を読む。"""
    out = {}
    with open(f"results/halton_{vit}384_layercache_bf16_sweep.txt") as f:
        for ln in f:
            p = ln.split()
            if p and p[0] in ("baseline",) + tuple(CONFIGS) and not ln.startswith("#"):
                out[p[0]] = float(p[1])
    return out


def read_speedup(vit):
    """latency csv から config -> (端到端, transformer-only) 加速比を読む。"""
    out = {}
    with open(f"results/latency_{vit}384_layercache_bf16.csv") as f:
        body = [l for l in f.read().splitlines() if not l.startswith('"#')]
    for r in csv.DictReader(body):
        key = ("baseline" if r["method"] == "baseline"
               else "norefresh" if r["refresh_n"] == "0"
               else "N" + r["refresh_n"])
        out[key] = (float(r["speedup_total"]), float(r["speedup_vit"]))
    return out


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=INK_MUTED, alpha=.22, linewidth=.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
        ax.spines[side].set_linewidth(.8)
    ax.tick_params(colors=INK_SOFT, labelsize=8.5, length=3, width=.8)


fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.2, 3.9))
fig.patch.set_facecolor(SURFACE)

ZOOM = [c for c in CONFIGS if c not in ("N1", "norefresh")]
zoom_x = []

for vit, label, color, marker, off_a, off_b in MODELS:
    fid, spd = read_fid(vit), read_speedup(vit)
    base_fid = fid["baseline"]

    # ── (a) 絶対 FID: baseline(=N1) から norefresh まで全域 ──
    xs = [spd[c][0] for c in CONFIGS]
    ys = [fid[c] for c in CONFIGS]
    axA.plot(xs, ys, "-", color=color, linewidth=2, zorder=3,
             marker=marker, markersize=6, markerfacecolor=color,
             markeredgecolor=SURFACE, markeredgewidth=1.2, label=label)
    dx, dy, ha = off_a
    for c, x, y in zip(CONFIGS, xs, ys):
        if c in LABELLED_A:
            axA.annotate(PRETTY[c], (x, y), textcoords="offset points",
                         xytext=(dx, dy), ha=ha, fontsize=7.5, color=INK_SOFT)

    # ── (b) ΔFID: 実用域 N=2..6 のみ ──
    xz = [spd[c][0] for c in ZOOM]
    yz = [fid[c] - base_fid for c in ZOOM]
    zoom_x += xz
    axB.plot(xz, yz, "-", color=color, linewidth=2, zorder=3,
             marker=marker, markersize=6, markerfacecolor=color,
             markeredgecolor=SURFACE, markeredgewidth=1.2, label=label)
    for c, x, y in zip(ZOOM, xz, yz):
        if (vit, c) in LABEL_OFF_B:
            dx, dy, ha = LABEL_OFF_B[(vit, c)]
            axB.annotate(PRETTY[c], (x, y), textcoords="offset points",
                         xytext=(dx, dy), ha=ha, fontsize=7.5, color=INK_SOFT)

# (a) の中に「(b) で拡大している範囲」を薄く示して 2 枚を繋ぐ
axA.axvspan(min(zoom_x) - .015, max(zoom_x) + .015, color=INK_MUTED, alpha=.10,
            zorder=0, linewidth=0)
axA.annotate("range of (b)", (min(zoom_x) - .01, axA.get_ylim()[1]),
             textcoords="offset points", xytext=(4, -11), ha="left",
             fontsize=7, color=INK_MUTED)

for ax in (axA, axB):
    style_axes(ax)
    ax.set_xlabel("End-to-end speedup  (higher is better)", fontsize=9, color=INK_SOFT)

axA.set_ylabel("FID  (lower is better)", fontsize=9, color=INK_SOFT)
axA.set_title("(a) Full range: refresh is not optional", fontsize=10,
              color=INK, loc="left", pad=10)
axB.set_ylabel("ΔFID vs. baseline", fontsize=9, color=INK_SOFT)
axB.set_title("(b) Operating range: smaller models degrade far less",
              fontsize=10, color=INK, loc="left", pad=10)
axB.set_yscale("log")
axB.set_ylim(.004, 2.6)
axB.set_yticks([.01, .03, .1, .3, 1])
axB.set_yticklabels(["0.01", "0.03", "0.1", "0.3", "1.0"])
axB.annotate("N = 2 → 6 along each curve", (.03, .965), xycoords="axes fraction",
             ha="left", va="top", fontsize=7.5, color=INK_MUTED)

lo, hi = axA.get_ylim()          # (a) 側だけ直接ラベル分の余白を足す
axA.set_ylim(lo - (hi - lo) * .04, hi + (hi - lo) * .08)

leg = axB.legend(frameon=False, fontsize=8.5, loc="lower right",
                 handlelength=1.8, labelspacing=.45)
for t in leg.get_texts():
    t.set_color(INK)

fig.text(.5, -.03,
         "ImageNet 384, 50k samples, bfloat16, 32 Halton steps.  "
         "Speedup: single A6000, batch 8, end-to-end (incl. VQGAN decode); "
         "transformer-only values are higher (e.g. large N=2: 1.335x vs 1.296x).",
         ha="center", fontsize=7.2, color=INK_SOFT)

fig.tight_layout()
os.makedirs("statics", exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"statics/layercache_bf16_pareto.{ext}", dpi=200,
                bbox_inches="tight", facecolor=SURFACE)
print("saved -> statics/layercache_bf16_pareto.{pdf,png}")
