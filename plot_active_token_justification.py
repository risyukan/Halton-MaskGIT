"""
Figure: why the active-token set = {current-step unmask} ∪ {previous-step unmask}.

Reads statics/ffn_delta/ffn_delta_stability.npz and shows that a committed token's
FFN update δ_l(t) is *unstable* only for exactly one step after it is released
(token age == 1), and becomes ~frozen from age >= 2 onward. Hence only two token
sets carry fresh FFN signal and must be recomputed each step:
  - current-step unmask  U_t   (age 0, being decided now — no cache exists)
  - previous-step unmask U_{t-1} (age 1 — its FFN output still swings hugely)
Everything at age >= 2 reuses the cached δ with negligible error.

Output: statics/ffn_delta/active_token_justification.{png,pdf}
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ---- design tokens (dataviz reference palette, light surface) ----
C_ACTIVE   = "#e34948"   # red  — unstable, must recompute (age 0/1 = active set)
C_CACHE    = "#2a78d6"   # blue — stable, cacheable (age >= 2)
INK        = "#0b0b0b"
INK_2      = "#52514e"
MUTED      = "#898781"
GRID       = "#e1e0d9"
SURFACE    = "#fcfcfb"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": "#c3c2b7",
    "axes.linewidth": 0.9,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
})

d = np.load("statics/ffn_delta/ffn_delta_stability.npz")
cos_age   = d["cos_by_age"]
drift_age = d["drift_by_age"]

ages = np.arange(len(cos_age))
valid = ~np.isnan(cos_age) & (ages >= 1)
ages   = ages[valid]
cos    = cos_age[valid]
drift  = drift_age[valid]

# plateau (cacheable) reference = mean over age >= 2
plateau_cos   = np.nanmean(cos[ages >= 2])
plateau_drift = np.nanmean(drift[ages >= 2])

# color per bar: age 1 = active/unstable, age >= 2 = cacheable
def bar_colors(a):
    return [C_ACTIVE if x == 1 else C_CACHE for x in a]

fig, (axC, axD) = plt.subplots(1, 2, figsize=(13, 5.0))

# ---------------- Panel A: cosine similarity vs prev step ----------------
axC.axhspan(plateau_cos - 0.004, 1.0, color=C_CACHE, alpha=0.06, zorder=0)
axC.axhline(plateau_cos, color=C_CACHE, lw=1.2, ls="--", zorder=1)
axC.bar(ages, cos, color=bar_colors(ages), width=0.78,
        edgecolor=SURFACE, linewidth=1.2, zorder=3)

axC.text(ages.max(), plateau_cos, f"  cos ≈ {plateau_cos:.2f}",
         color=C_CACHE, fontsize=9.5, fontweight="bold", va="bottom", ha="right")
axC.annotate(f"{cos[0]:.2f}", (1, cos[0]), textcoords="offset points",
             xytext=(0, 6), ha="center", color=C_ACTIVE, fontweight="bold")
axC.annotate("age 1: FFN output\nflips — must recompute",
             xy=(1, cos[0]), xytext=(4.2, 0.63),
             color=C_ACTIVE, fontsize=10, fontweight="bold", va="center",
             arrowprops=dict(arrowstyle="-|>", color=C_ACTIVE, lw=1.6,
                             connectionstyle="arc3,rad=-0.2"))

axC.set_ylim(0.45, 1.02)
axC.set_xlim(0.3, ages.max() + 0.7)
axC.set_ylabel(r"cos$\,(\delta_l(t),\,\delta_l(t{-}1))$", color=INK)
axC.set_xlabel("token age  (steps since first unmasked)", color=INK_2)
axC.set_title("FFN update direction vs. previous step", color=INK, fontweight="bold", loc="left")
axC.grid(axis="y", color=GRID, lw=0.8, zorder=0)
axC.set_axisbelow(True)
for s in ("top", "right"):
    axC.spines[s].set_visible(False)

# ---------------- Panel B: relative drift vs prev step ----------------
axD.axhspan(0.0, plateau_drift + 0.02, color=C_CACHE, alpha=0.06, zorder=0)
axD.axhline(plateau_drift, color=C_CACHE, lw=1.2, ls="--", zorder=1)
axD.bar(ages, drift, color=bar_colors(ages), width=0.78,
        edgecolor=SURFACE, linewidth=1.2, zorder=3)

axD.text(2.4, plateau_drift + 0.05,
         f"cacheable plateau  drift ≈ {plateau_drift:.2f}",
         color=C_CACHE, fontsize=10, fontweight="bold", va="bottom")
axD.annotate(f"{drift[0]:.2f}", (1, drift[0]), textcoords="offset points",
             xytext=(0, 6), ha="center", color=C_ACTIVE, fontweight="bold")
axD.annotate("age 1: ~6× the\ncached error",
             xy=(1, drift[0]), xytext=(4.2, 0.72),
             color=C_ACTIVE, fontsize=10, fontweight="bold", va="center",
             arrowprops=dict(arrowstyle="-|>", color=C_ACTIVE, lw=1.6,
                             connectionstyle="arc3,rad=0.2"))

axD.set_ylim(0.0, 0.95)
axD.set_xlim(0.3, ages.max() + 0.7)
axD.set_ylabel(r"relative drift  $\|\Delta\delta\|_2 / \|\delta(t)\|_2$", color=INK)
axD.set_xlabel("token age  (steps since first unmasked)", color=INK_2)
axD.set_title("FFN update magnitude change vs. previous step", color=INK, fontweight="bold", loc="left")
axD.grid(axis="y", color=GRID, lw=0.8, zorder=0)
axD.set_axisbelow(True)
for s in ("top", "right"):
    axD.spines[s].set_visible(False)

# ---------------- shared legend ----------------
from matplotlib.patches import Patch
handles = [
    Patch(facecolor=C_ACTIVE, label=r"active $\;U_{t-1}$ (age 1) — recompute"),
    Patch(facecolor=C_CACHE,  label=r"age $\geq 2$ — reuse cached $\delta$"),
]
# fix the escaped dash in labels
handles[0].set_label(r"active  $U_{t-1}$ (age 1) — recompute")
handles[1].set_label(r"age $\geq 2$ — reuse cached $\delta$")
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, -0.02), fontsize=10.5)

fig.suptitle(
    "A committed token's FFN update is unstable for exactly one step, then freezes",
    fontsize=13.5, fontweight="bold", color=INK, x=0.02, ha="left", y=0.99)

fig.tight_layout(rect=(0, 0.04, 1, 0.96))

os.makedirs("statics/ffn_delta", exist_ok=True)
for ext in ("png", "pdf"):
    p = f"statics/ffn_delta/active_token_justification.{ext}"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print("[plot] saved", p)

print(f"[stats] age1 cos={cos[0]:.3f} drift={drift[0]:.3f} | "
      f"plateau cos={plateau_cos:.3f} drift={plateau_drift:.3f} | "
      f"drift ratio age1/plateau = {drift[0]/plateau_drift:.1f}x")
