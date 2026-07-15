"""Re-plot the FFN-delta (layer × step) heatmap over ALL inactive tokens.

The original ffn_delta_heatmap.png aggregates only *committed* inactive tokens,
but the partial-update FFN cache (Network/transformer.py, Block.forward) applies
the cached delta to *every* inactive position — committed AND still-masked.
To make the motivation figure match what the cache actually does, we recompute
the same mean-cosine / mean-rel-drift heatmaps over the full inactive set
(inactive = ~active_at), reusing the raw per-token arrays in the saved npz so no
model inference is needed.

Outputs:
  statics/ffn_delta/ffn_delta_heatmap_all_inactive.png
  statics/ffn_delta/ffn_delta_heatmap_compare.png   (committed vs all-inactive)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Larger fonts for poster readability.
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
})

NPZ = "statics/ffn_delta/ffn_delta_stability.npz"
OUT_DIR = "statics/ffn_delta"


def grouped_mean(arr, mask, valid, axes):
    """Mean of `arr` over `axes`, restricted to mask & valid; NaN for empty groups.
    Mirrors aggregate_and_plot.grouped_mean in analyze_ffn_delta_stability.py."""
    m = mask & valid
    a_ = np.where(m, arr, 0.0)
    s = a_.sum(axis=axes)
    c = m.sum(axis=axes)
    out = s / c.clip(min=1)
    out[c == 0] = np.nan
    return out


def main():
    d = np.load(NPZ)
    cos = d["cos"].astype(np.float32)        # (T, L, B, N)
    drift = d["drift"].astype(np.float32)    # (T, L, B, N)
    active_at = d["active_at"]               # (T, B, N) bool
    commit_step = d["commit_step"]           # (B, N)
    T, L, B, N = cos.shape

    step_idx = np.arange(T)[:, None, None]            # (T, 1, 1)
    commit_b = commit_step[None, :, :]                # (1, B, N)
    in_M_t = (commit_b >= 0) & (commit_b <= step_idx)  # (T, B, N) already committed

    inactive = ~active_at                             # (T, B, N) ALL inactive
    inactive_committed = inactive & in_M_t            # for side-by-side reference

    def bcast(m):
        return np.broadcast_to(m[:, None, :, :], cos.shape)

    valid = ~np.isnan(cos)

    # (layer × step) heatmaps — note: collapse over B, N only (axes=(2, 3)).
    heat_cos_all = grouped_mean(cos,   bcast(inactive),           valid, axes=(2, 3))  # (T, L)
    heat_drf_all = grouped_mean(drift, bcast(inactive),           valid, axes=(2, 3))
    heat_cos_cmt = grouped_mean(cos,   bcast(inactive_committed), valid, axes=(2, 3))
    heat_drf_cmt = grouped_mean(drift, bcast(inactive_committed), valid, axes=(2, 3))

    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- standalone all-inactive heatmap ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    im = axes[0].imshow(heat_cos_all.T, aspect="auto", origin="lower", cmap="viridis")
    axes[0].set_xlabel("step (t)"); axes[0].set_ylabel("layer")
    axes[0].set_title("mean cosine — all inactive tokens")
    fig.colorbar(im, ax=axes[0]).ax.tick_params(labelsize=14)

    im = axes[1].imshow(heat_drf_all.T, aspect="auto", origin="lower", cmap="magma")
    axes[1].set_xlabel("step (t)"); axes[1].set_ylabel("layer")
    axes[1].set_title("mean rel-drift — all inactive tokens")
    fig.colorbar(im, ax=axes[1]).ax.tick_params(labelsize=14)

    plt.tight_layout()
    p1 = os.path.join(OUT_DIR, "ffn_delta_heatmap_all_inactive.png")
    plt.savefig(p1, dpi=120); plt.close(fig)
    print("[plot] saved", p1)

    # ---- 2×2 compare: committed (top) vs all-inactive (bottom), shared scales ----
    cos_vmin = np.nanmin([heat_cos_cmt, heat_cos_all])
    cos_vmax = np.nanmax([heat_cos_cmt, heat_cos_all])
    drf_vmin = np.nanmin([heat_drf_cmt, heat_drf_all])
    drf_vmax = np.nanmax([heat_drf_cmt, heat_drf_all])

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    panels = [
        (axes[0, 0], heat_cos_cmt, "viridis", cos_vmin, cos_vmax, "mean cosine — committed inactive"),
        (axes[0, 1], heat_drf_cmt, "magma",   drf_vmin, drf_vmax, "mean rel-drift — committed inactive"),
        (axes[1, 0], heat_cos_all, "viridis", cos_vmin, cos_vmax, "mean cosine — all inactive"),
        (axes[1, 1], heat_drf_all, "magma",   drf_vmin, drf_vmax, "mean rel-drift — all inactive"),
    ]
    for ax, hm, cmap, vmin, vmax, title in panels:
        im = ax.imshow(hm.T, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xlabel("step (t)"); ax.set_ylabel("layer"); ax.set_title(title)
        fig.colorbar(im, ax=ax).ax.tick_params(labelsize=14)

    plt.tight_layout()
    p2 = os.path.join(OUT_DIR, "ffn_delta_heatmap_compare.png")
    plt.savefig(p2, dpi=120); plt.close(fig)
    print("[plot] saved", p2)

    # ---- text summary ----
    def rng(name, arr):
        print(f"{name:34s} cos∈[{np.nanmin(arr[0]):.4f},{np.nanmax(arr[0]):.4f}]  "
              f"drift∈[{np.nanmin(arr[1]):.4f},{np.nanmax(arr[1]):.4f}]")
    print("\n=== heatmap value ranges ===")
    rng("committed inactive", (heat_cos_cmt, heat_drf_cmt))
    rng("all inactive",       (heat_cos_all, heat_drf_all))
    # post-warmup (t>=7) means, the region the cache actually exploits
    sl = slice(7, T)
    print(f"\npost-warmup (t>=7) mean cosine: committed={np.nanmean(heat_cos_cmt[sl]):.4f}  "
          f"all={np.nanmean(heat_cos_all[sl]):.4f}")
    print(f"post-warmup (t>=7) mean drift : committed={np.nanmean(heat_drf_cmt[sl]):.4f}  "
          f"all={np.nanmean(heat_drf_all[sl]):.4f}")


if __name__ == "__main__":
    main()
