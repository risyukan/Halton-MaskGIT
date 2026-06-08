"""
Post-hoc slice of kv_drift.npz by token age, with focus on age >= 2.

age = t - commit_step[n]
  age=1 → just-committed last step. Its input id at t-1 was mask_value,
          at t is the committed id, so K/V drift at layer 0 is *expected* to
          be large for this group. The existing partial-update scheme already
          re-computes age<=1 tokens (active = U_{t-1} ∪ U_t), so these tokens
          would never sit in a hypothetical KV cache.
  age>=2 → input id is fixed at both t-1 and t. Layer-0 K/V drift in proj=1
          mode should be ~0. Deeper layers drift only via context.

This script answers: in the actual cache-target population (age >= 2 inactive
committed), how much do K/V drift?
"""
import argparse
import os
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", type=str, default="statics/kv_drift/kv_drift.npz")
    p.add_argument("--gate-layer-min", type=int, default=3)
    p.add_argument("--gate-layer-max", type=int, default=21)
    p.add_argument("--gate-step-min",  type=int, default=5)
    p.add_argument("--gate-step-max",  type=int, default=30)
    p.add_argument("--out-dir", type=str, default="statics/kv_drift")
    return p.parse_args()


def main():
    args = parse_args()
    data = np.load(args.npz, allow_pickle=False)
    drift_k = data["drift_k"].astype(np.float32)   # (T, L, B, N)
    drift_v = data["drift_v"].astype(np.float32)
    cos_k   = data["cos_k"].astype(np.float32)
    cos_v   = data["cos_v"].astype(np.float32)
    commit_step = data["commit_step"]              # (B, N)
    active_at   = data["active_at"]                # (T, B, N)
    T, L, B, N = drift_k.shape

    # age[t,b,n] = t - commit_step[b,n], undefined where commit_step<0
    step_idx = np.arange(T)[:, None, None]
    commit_b = commit_step[None, :, :]
    age = step_idx - commit_b
    in_M_t = (commit_b >= 0) & (commit_b <= step_idx)
    inactive = ~active_at

    valid = ~np.isnan(drift_k)
    smin, smax = args.gate_step_min, args.gate_step_max
    lmin, lmax = args.gate_layer_min, args.gate_layer_max

    def gate_stats(mask_3d, arr):
        """mask_3d: (T, B, N) — token-level. arr: (T, L, B, N)."""
        m4 = np.broadcast_to(mask_3d[:, None, :, :], arr.shape) & valid
        m_gate = m4[smin:smax + 1, lmin:lmax + 1]
        a_gate = arr[smin:smax + 1, lmin:lmax + 1]
        if not m_gate.any():
            return float("nan"), float("nan"), 0
        sub = a_gate[m_gate]
        return float(sub.mean()), float(sub.max()), int(m_gate.sum())

    def layer0_stats(mask_3d, arr):
        m4 = np.broadcast_to(mask_3d[:, None, :, :], arr.shape) & valid
        m_l0 = m4[:, 0]            # (T, B, N)
        a_l0 = arr[:, 0]
        if not m_l0.any():
            return float("nan")
        return float(a_l0[m_l0].mean())

    rows = []
    age_buckets = [
        ("all committed-inactive (age >= 1)", inactive & in_M_t & (age >= 1)),
        ("age == 1 (just committed)",         inactive & in_M_t & (age == 1)),
        ("age == 2",                          inactive & in_M_t & (age == 2)),
        ("age == 3",                          inactive & in_M_t & (age == 3)),
        ("age >= 2 (real cache target)",      inactive & in_M_t & (age >= 2)),
        ("age >= 3",                          inactive & in_M_t & (age >= 3)),
        ("age >= 5",                          inactive & in_M_t & (age >= 5)),
        ("still-masked (NOT a cache target)", inactive & ~in_M_t),
    ]
    print(f"{'group':<42}  {'layer0_K':>9}  {'layer0_V':>9}  "
          f"{'gate_K_mean':>11}  {'gate_K_max':>10}  "
          f"{'gate_V_mean':>11}  {'gate_V_max':>10}  {'n_obs':>10}")
    print("-" * 130)
    for label, m3 in age_buckets:
        l0k = layer0_stats(m3, drift_k)
        l0v = layer0_stats(m3, drift_v)
        gmk, gxk, nobs = gate_stats(m3, drift_k)
        gmv, gxv, _    = gate_stats(m3, drift_v)
        print(f"{label:<42}  {l0k:>9.4f}  {l0v:>9.4f}  "
              f"{gmk:>11.4f}  {gxk:>10.4f}  {gmv:>11.4f}  {gxv:>10.4f}  {nobs:>10d}")
        rows.append((label, l0k, l0v, gmk, gxk, gmv, gxv, nobs))

    # --- Plot: drift heatmap (layer x step) restricted to age>=2 ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    target_mask = inactive & in_M_t & (age >= 2)  # (T, B, N)
    valid_full = ~np.isnan(drift_k)

    def heat(arr):
        m4 = np.broadcast_to(target_mask[:, None, :, :], arr.shape) & valid_full
        s = np.where(m4, arr, 0.0).sum(axis=(2, 3))
        c = m4.sum(axis=(2, 3)).clip(min=1)
        out = s / c
        out[m4.sum(axis=(2, 3)) == 0] = np.nan
        return out  # (T, L)

    heat_dk = heat(drift_k)
    heat_dv = heat(drift_v)
    heat_ck = heat(cos_k)
    heat_cv = heat(cos_v)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, h, title, cmap in (
        (axes[0, 0], heat_ck, "K cosine — age>=2 committed", "viridis"),
        (axes[0, 1], heat_dk, "K rel-drift — age>=2 committed", "magma"),
        (axes[1, 0], heat_cv, "V cosine — age>=2 committed", "viridis"),
        (axes[1, 1], heat_dv, "V rel-drift — age>=2 committed", "magma"),
    ):
        im = ax.imshow(h.T, aspect="auto", origin="lower", cmap=cmap)
        rect = Rectangle((smin - 0.5, lmin - 0.5),
                         smax - smin + 1, lmax - lmin + 1,
                         linewidth=1.5, edgecolor="white", facecolor="none",
                         linestyle="--")
        ax.add_patch(rect)
        ax.set_xlabel("step (t)"); ax.set_ylabel("layer")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
    plt.tight_layout()
    p_out = os.path.join(args.out_dir, "kv_heatmap_age2plus.png")
    plt.savefig(p_out, dpi=120); plt.close(fig)
    print(f"\n[plot] saved {p_out}")


if __name__ == "__main__":
    main()
