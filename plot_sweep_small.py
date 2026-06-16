"""Plot the small-384 partial-update cache-refresh sweep.

Left  Y axis : FID (lower is better)
Right Y axis : theoretical speedup based on FLOPs reduction (higher is better)
X axis       : cache refresh interval N (categorical, includes the "no refresh"
               case rendered as ∞)
"""
import os
import matplotlib.pyplot as plt

# --- FID data (from results/halton_small384_cfg10_partialupdate_sweep.txt) ---
# N=1 is included as the left-anchor point: every step is a refresh, so it is
# numerically equivalent to the full-update baseline (FID 6.101, speedup 1.000×).
# ∞ uses the txt partial_no_refresh value 7.5441 for consistency with the
# other points.
N_LABELS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '∞']
X        = list(range(10))               # equal spacing on x-axis
FID      = [6.101, 6.1334, 6.0542, 6.0163, 6.0311,
            6.0416, 6.0186, 6.096, 6.109, 7.5441]

# --- theoretical FLOPs / speedup model ---
# Per-block FLOP breakdown (Small model, d=512, seq=576):
#   Attention proj 4*s*d^2 + matmul 2*s^2*d = 0.77 GFLOPs
#   FFN  8*s*d^2 = 1.21 GFLOPs
# -> FFN占 1.21 / (0.77+1.21) ≈ 0.56 of per-block compute
F_FFN_FRAC = 0.56

TOTAL_STEPS     = 32
GATED_STEPS     = 26                  # step indices 5..30 inclusive
TOTAL_LAYERS    = 12
GATED_LAYERS    = 7                   # layer indices 3..9 inclusive
SEQ_LEN         = 576                 # 24x24 token grid
N_ACTIVE        = 36                  # |U_{t-1} ∪ U_t| ≈ 2 × 576/32

ACTIVE_FRAC          = N_ACTIVE / SEQ_LEN                    # ≈ 0.0625
PARTIAL_BLOCK_RATIO  = (1 - F_FFN_FRAC) + F_FFN_FRAC * ACTIVE_FRAC
# Step-level: gated layers shrink to PARTIAL_BLOCK_RATIO of a full block, the
# rest of the layers stay full -> ratio of one partial step vs a full step:
PARTIAL_STEP_RATIO = (
    (TOTAL_LAYERS - GATED_LAYERS) / TOTAL_LAYERS
    + GATED_LAYERS / TOTAL_LAYERS * PARTIAL_BLOCK_RATIO
)


def theoretical_speedup(refresh_n):
    """N=0 -> never refresh (pure partial); N>=1 -> every N-th gated step is full."""
    if refresh_n == 0:
        n_refresh = 0
    else:
        n_refresh = (GATED_STEPS + refresh_n - 1) // refresh_n   # ceil(26/N)
    n_partial = GATED_STEPS - n_refresh
    non_gated = TOTAL_STEPS - GATED_STEPS                        # always full
    cost = n_partial * PARTIAL_STEP_RATIO + n_refresh * 1.0 + non_gated * 1.0
    baseline_cost = TOTAL_STEPS * 1.0
    return baseline_cost / cost


# refresh interval per point: N=1..9, then 0 (=∞, never refresh)
REFRESH_N = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
SPEEDUP   = [theoretical_speedup(n) for n in REFRESH_N]
for lbl, spd in zip(N_LABELS, SPEEDUP):
    print(f'  N={lbl:>2}  speedup {spd:.3f}×')

BASELINE_FID     = 6.101
BASELINE_LABEL   = f'baseline (full FFN) FID = {BASELINE_FID:.3f}'
PARETO_BUDGET    = BASELINE_FID + 0.5
PARETO_LABEL     = f'+0.5 FID budget = {PARETO_BUDGET:.3f}'

OUT_PATH = os.path.join(os.path.dirname(__file__),
                        'results', 'sweep_fid_vs_speedup_small.png')

# --- plot ---
fig, ax_fid = plt.subplots(figsize=(7.5, 4.8))
color_fid = '#c0392b'
color_spd = '#2c6fbb'

# FID line (left axis)
ax_fid.plot(X, FID, color=color_fid, marker='o', markersize=8,
            linewidth=2, label='FID', zorder=3)
ax_fid.set_xlabel('Cache refresh interval $N$', fontsize=12)
ax_fid.set_ylabel('FID  ↓', color=color_fid, fontsize=12)
ax_fid.tick_params(axis='y', labelcolor=color_fid)
ax_fid.set_xticks(X)
ax_fid.set_xticklabels(N_LABELS, fontsize=11)
ax_fid.grid(True, alpha=0.3, linestyle=':')

# baseline FID horizontal lines
ax_fid.axhline(BASELINE_FID, color='gray', linestyle='--', linewidth=1,
               alpha=0.8, label=BASELINE_LABEL)
ax_fid.axhline(PARETO_BUDGET, color='gray', linestyle=':', linewidth=1,
               alpha=0.6, label=PARETO_LABEL)

# annotate FID values
for x, y in zip(X, FID):
    ax_fid.annotate(f'{y:.3f}', (x, y),
                    textcoords='offset points', xytext=(0, 12),
                    ha='center', fontsize=10, color=color_fid)

# Speedup line (right axis)
ax_spd = ax_fid.twinx()
ax_spd.plot(X, SPEEDUP, color=color_spd, marker='s', markersize=8,
            linewidth=2, linestyle='--', label='Theoretical speedup (FLOPs)', zorder=3)
ax_spd.set_ylabel('Theoretical FLOPs speedup ×  ↑', color=color_spd, fontsize=12)
ax_spd.tick_params(axis='y', labelcolor=color_spd)

# annotate speedup values (to the right of each point, vertically centered)
for x, y in zip(X, SPEEDUP):
    ax_spd.annotate(f'{y:.3f}×', (x, y),
                    textcoords='offset points', xytext=(10, 0),
                    ha='left', va='center', fontsize=10, color=color_spd)

# y-axis ranges - make FID axis breathable, especially around N=∞
ax_fid.set_ylim(5.8, 7.8)
ax_spd.set_ylim(0.95, 1.45)

# combined legend (top-left)
h1, l1 = ax_fid.get_legend_handles_labels()
h2, l2 = ax_spd.get_legend_handles_labels()
ax_fid.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=9, framealpha=0.92)

plt.title('FID vs Cache Refresh Interval — H-MaskGIT-S 384 (fp32, cfg 1.0)',
          fontsize=12, pad=12)
fig.tight_layout()

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
plt.savefig(OUT_PATH, dpi=160, bbox_inches='tight')
print(f'saved -> {OUT_PATH}')
