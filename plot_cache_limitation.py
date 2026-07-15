"""Schematic for the poster's 'Limitations of FFN Delta cache in MaskGIT' panel.

Left  — Halton-MaskGIT: the unmask order is precomputed (Halton sequence), so the
        active-token set at every step is known BEFORE the forward pass. The FFN
        can therefore run on active tokens only and reuse cached deltas elsewhere.

Right — Vanilla (confidence-based) MaskGIT: the tokens unmasked this step are the
        current top-k by model confidence, which only exists AFTER a full forward.
        So 'run FFN on active only' depends on knowing the active set, which depends
        on a full forward over ALL tokens — a circular dependency that defeats the
        cache.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

GREEN = "#2e7d32"
GREEN_BG = "#e6f4ea"
RED = "#c62828"
RED_BG = "#fdecea"
GREY = "#444444"


def box(ax, xy, w, h, text, fc, ec, fontsize=15, fontweight="normal", tc="black"):
    x, y = xy
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle="round,pad=0.012,rounding_size=0.03",
                       linewidth=2.2, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight=fontweight, color=tc, zorder=3)


def arrow(ax, p0, p1, color=GREY, rad=0.0, lw=2.4, ls="-"):
    ax.annotate("", xy=p1, xytext=p0, zorder=1,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                linestyle=ls,
                                connectionstyle=f"arc3,rad={rad}",
                                shrinkA=14, shrinkB=14, mutation_scale=22))


def main():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 7.5))
    for ax in (axL, axR):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ───────────────────────── LEFT: Halton (works) ─────────────────────────
    axL.set_title("Halton-MaskGIT — schedule is fixed",
                  fontsize=20, fontweight="bold", color=GREEN, pad=14)
    box(axL, (0.5, 0.86), 0.78, 0.15,
        "Unmask order precomputed\n(Halton quasi-random sequence)",
        GREEN_BG, GREEN, fontsize=15, fontweight="bold")
    box(axL, (0.5, 0.55), 0.78, 0.15,
        "Active tokens at step $t$ are known\nBEFORE the forward pass",
        "white", GREEN, fontsize=15)
    box(axL, (0.5, 0.22), 0.78, 0.16,
        "FFN runs on active tokens only\n+ reuse cached $\\Delta$ for inactive",
        GREEN_BG, GREEN, fontsize=15, fontweight="bold")
    arrow(axL, (0.5, 0.785), (0.5, 0.625), color=GREEN)
    arrow(axL, (0.5, 0.475), (0.5, 0.30), color=GREEN)
    axL.text(0.5, 0.05, "✓  cache is well-defined",
             ha="center", va="center", fontsize=17, fontweight="bold", color=GREEN)

    # ──────────────── RIGHT: vanilla MaskGIT (circular dep) ──────────────────
    axR.set_title("Vanilla MaskGIT — unmask by confidence",
                  fontsize=20, fontweight="bold", color=RED, pad=14)
    # four nodes of the dependency cycle
    A = (0.5, 0.88)   # goal
    B = (0.78, 0.52)  # which are active?
    C = (0.5, 0.16)   # need confidence of all
    D = (0.22, 0.52)  # need full forward
    box(axR, A, 0.56, 0.14, "Goal: run FFN on\nactive tokens only",
        RED_BG, RED, fontsize=14, fontweight="bold")
    box(axR, B, 0.36, 0.16, "needs: which tokens\nare active this step?",
        "white", GREY, fontsize=12.5)
    box(axR, C, 0.52, 0.14, "needs: confidence\nscores of ALL tokens",
        "white", GREY, fontsize=12.5)
    box(axR, D, 0.36, 0.16, "needs: FULL forward\n(FFN on every token)",
        RED_BG, RED, fontsize=12.5, fontweight="bold")
    # cycle arrows (clockwise)
    arrow(axR, A, B, color=GREY, rad=-0.25)
    arrow(axR, B, C, color=GREY, rad=-0.25)
    arrow(axR, C, D, color=GREY, rad=-0.25)
    arrow(axR, D, A, color=RED, rad=-0.25, ls=(0, (5, 3)))
    # contradiction callout in the middle
    axR.text(0.5, 0.52, "circular\ndependency", ha="center", va="center",
             fontsize=16, fontweight="bold", color=RED,
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=RED, lw=2))
    axR.text(0.5, 0.025,
             "✗  active set unknown until after a full forward",
             ha="center", va="center", fontsize=16, fontweight="bold", color=RED)

    plt.tight_layout()
    out = "statics/ffn_delta/cache_limitation_maskgit.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("[plot] saved", out)


if __name__ == "__main__":
    main()
