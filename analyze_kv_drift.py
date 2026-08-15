"""
Quantify K/V drift for committed tokens across Halton-MaskGIT decoding steps.

For vanilla (full-token-update) inference, at every step t and every transformer
layer l we record xk_l(t) and xv_l(t) (post-QKNorm projections, before
head-split / attention) per token, then for each inactive token we compute:

    cos_l(t)   = cosine( x_l(t), x_l(t-1) )      # over feature dim
    drift_l(t) = || x_l(t) - x_l(t-1) ||_2 / || x_l(t) ||_2

separately for x = K and x = V.

This answers the question: "if we cache K/V of committed tokens between
iterations of MaskGIT parallel decoding, how much does the truth drift?"

Two inactive sub-populations are tracked separately (same convention as
analyze_ffn_delta_stability.py):
  (a) committed-before-t: token id is fixed — *this is the KV-cache target*
  (b) still-masked:       token is mask_value at both t and t-1

Sanity check: at layer 0 with proj=1, committed tokens have a fully fixed
input (token_emb + pos_emb), and class-conditioned modulation is also
step-independent, so K/V drift should be ~0 there. Any non-zero number at
layer 0 in proj=1 mode = a bug.

Outputs:
  statics/kv_drift/kv_drift.npz   raw per-token cos / drift for K and V
  statics/kv_drift/kv_drift.png   8-panel (K row + V row) by-layer / step / age
  statics/kv_drift/kv_heatmap.png (layer × step) heatmaps for K and V
"""

import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Utils.utils import load_args_from_file
from Trainer.cls_trainer import MaskGIT
from Sampler.halton_sampler import HaltonSampler
from Network.transformer import Attention, Block, modulate
import torch.nn.functional as F


_K_BUFFER: list = []
_V_BUFFER: list = []


def _patched_attn_forward(self, x, mask=None, active_idx=None):
    """Mirror Attention.forward (full-update branch) and capture xk, xv post-QKNorm.

    Only the active_idx=None branch is used by this analyzer — we never pass
    active_idx here, so the partial-update branch is intentionally omitted.
    """
    assert active_idx is None, "kv-drift analyzer only runs vanilla full-update"
    b, h_w, _ = x.shape
    xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
    xq, xk = self.qk_norm(xq, xk, xv)

    # Capture in fp32 to keep the drift signal clean (bf16 noise floor is ~1e-3,
    # which is the same order as deep-layer drifts we want to resolve).
    _K_BUFFER.append(xk.detach().to(torch.float32).cpu())
    _V_BUFFER.append(xv.detach().to(torch.float32).cpu())

    xq = xq.view(b, h_w, self.n_local_heads, self.head_dim)
    xk = xk.view(b, h_w, self.n_local_heads, self.head_dim)
    xv = xv.view(b, h_w, self.n_local_heads, self.head_dim)
    xq, xk, xv = (t.transpose(1, 2) for t in (xq, xk, xv))
    if self.flash:
        if mask is not None:
            mask = mask.view(b, 1, 1, h_w)
        output = F.scaled_dot_product_attention(
            xq, xk, xv, mask,
            dropout_p=self.dropout if self.training else 0.,
        )
    else:
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, xv)
    output = output.transpose(1, 2).contiguous().view(b, h_w, -1)
    proj = self.wo(output)
    return proj


def patch_attention(transformer):
    for blk in transformer.layers:
        blk.attn.forward = _patched_attn_forward.__get__(blk.attn, Attention)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     type=str, default="Config/base_cls2img.yaml")
    p.add_argument("--vit-size",   type=str, default="large")
    p.add_argument("--img-size",   type=int, default=384)
    p.add_argument("--steps",      type=int, default=32)
    p.add_argument("--nb-sample",  type=int, default=4)
    p.add_argument("--cfg-w",      type=float, default=0.5,
                   help="match the partial-update sweep (cfg-w=0.5)")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--dtype",      type=str, default="float32",
                   help="float32 keeps the K/V drift signal clean; bf16 is much "
                        "faster but its noise floor (~1e-3) can swallow shallow "
                        "layer drifts.")
    p.add_argument("--out-dir",    type=str, default="statics/kv_drift")
    # Highlight box for the partial-update gate used elsewhere in the project.
    p.add_argument("--gate-layer-min", type=int, default=3)
    p.add_argument("--gate-layer-max", type=int, default=21)
    p.add_argument("--gate-step-min",  type=int, default=5)
    p.add_argument("--gate-step-max",  type=int, default=30)
    return p.parse_args()


def run_inference_capture(args_cli):
    """Run vanilla 32-step Halton sampling and stream K/V drift step by step.

    Returns:
        cos_k, drift_k, cos_v, drift_v: (steps, layers, b, n_tokens) float32
                                        (NaN at step 0; per-token, per-layer)
        commit_step:                    (b, n_tokens) int — step of first commit (-1 if never)
        active_at:                      (steps, b, n_tokens) bool — token in U_t at step t
    """
    cfg = load_args_from_file(args_cli.config)
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.vit_size = args_cli.vit_size
    cfg.img_size = args_cli.img_size
    cfg.compile  = False
    cfg.dtype    = args_cli.dtype
    cfg.resume   = True
    cfg.vit_folder = f"./saved_networks/ImageNet_{cfg.img_size}_{cfg.vit_size}.pth"
    cfg.data_folder = ""
    cfg.eval_folder = ""
    cfg.writer_log  = ""
    cfg.debug = True

    if args_cli.seed >= 0:
        torch.manual_seed(args_cli.seed)
        torch.cuda.manual_seed_all(args_cli.seed)
        np.random.seed(args_cli.seed)

    print(f"[init] loading MaskGIT (vit_size={cfg.vit_size}, img_size={cfg.img_size}, dtype={cfg.dtype})")
    trainer = MaskGIT(cfg)
    transformer = trainer.vit if not hasattr(trainer.vit, "module") else trainer.vit.module

    patch_attention(transformer.transformer)
    n_layers = len(transformer.transformer.layers)
    register = transformer.register
    print(f"[init] patched {n_layers} attention modules, register={register}")

    sampler = HaltonSampler(
        sm_temp_min=1, sm_temp_max=1.0, temp_pow=1, temp_warmup=1,
        w=args_cli.cfg_w, sched_pow=2, step=args_cli.steps,
        randomize=False, top_k=-1,
    )

    input_size = trainer.input_size
    nb = args_cli.nb_sample
    # Each CFG forward returns K/V for the cond half (rows [:nb]) and the
    # uncond half (rows [nb:2*nb]). Both halves go through cache in real
    # inference, so both are first-class observations here. Effective batch
    # for drift stats: B = 2*nb (cond stacked on top of uncond).
    B = 2 * nb
    n_tokens = input_size * input_size

    # Pre-compute U_t / M_t schedule (randomize=False matches the loop below).
    # Halton schedule is identical for cond/uncond, so we tile it across the
    # 2*nb effective batch.
    l_U_t, _ = sampler.compute_schedule(input_size, nb_sample=nb)
    active_at_cond = torch.stack(l_U_t).view(args_cli.steps, nb, n_tokens).bool().numpy()
    active_at = np.concatenate([active_at_cond, active_at_cond], axis=1)  # (T, 2*nb, N)

    commit_step = -np.ones((B, n_tokens), dtype=np.int64)
    for t in range(args_cli.steps):
        new = active_at[t] & (commit_step == -1)
        commit_step[new] = t

    demo_labels = [1, 7, 282, 604, 724, 179, 681, 850, 850]
    labels = torch.LongTensor(demo_labels[:nb]).to(cfg.device)

    trainer.vit.eval()
    drop = torch.ones(nb, dtype=torch.bool, device=cfg.device)
    code = torch.full((nb, input_size, input_size), cfg.mask_value,
                      dtype=torch.long, device=cfg.device)

    halton_mask = sampler.basic_halton_mask.clone().unsqueeze(0).expand(nb, n_tokens, 2)

    T = args_cli.steps
    # First B rows = cond, second B rows = uncond — kept separable for the
    # cond-vs-uncond sanity check at the end.
    cos_k   = np.full((T, n_layers, B, n_tokens), np.nan, dtype=np.float32)
    drift_k = np.full((T, n_layers, B, n_tokens), np.nan, dtype=np.float32)
    cos_v   = np.full((T, n_layers, B, n_tokens), np.nan, dtype=np.float32)
    drift_v = np.full((T, n_layers, B, n_tokens), np.nan, dtype=np.float32)

    prev_k = None
    prev_v = None

    with torch.no_grad():
        prev_r = 0
        for t in range(T):
            ratio = (t + 1) / T
            r = 1 - (torch.arccos(torch.tensor(ratio)) / (math.pi * 0.5))
            r = int(r * (input_size ** 2))
            r = max(t + 1, r)

            _u = halton_mask[:, prev_r:r]
            U_t = torch.zeros(nb, input_size, input_size, dtype=torch.bool)
            for i in range(nb):
                U_t[i, _u[i, :, 0], _u[i, :, 1]] = True

            _K_BUFFER.clear()
            _V_BUFFER.clear()
            with trainer.autocast:
                logit = trainer.vit(
                    torch.cat([code.clone(), code.clone()], dim=0),
                    torch.cat([labels, labels], dim=0),
                    torch.cat([~drop, drop], dim=0),
                    active_mask=None,
                )
            logit_c, logit_u = torch.chunk(logit, 2, dim=0)
            logit = (1 + sampler.w) * logit_c - sampler.w * logit_u

            assert len(_K_BUFFER) == n_layers, f"got {len(_K_BUFFER)} K, expected {n_layers}"
            assert len(_V_BUFFER) == n_layers, f"got {len(_V_BUFFER)} V, expected {n_layers}"

            # Buffers contain CFG-cat tensors (2*nb). Keep BOTH halves stacked
            # (cond on top of uncond) — real inference caches both, so drift
            # has to be acceptable on both. Registers dropped from the seq tail.
            curr_k, curr_v = [], []
            for k_buf, v_buf in zip(_K_BUFFER, _V_BUFFER):
                k_b = k_buf            # (2*nb, h_w, d)
                v_b = v_buf
                if register > 0:
                    k_b = k_b[:, :-register]
                    v_b = v_b[:, :-register]
                curr_k.append(k_b.contiguous())
                curr_v.append(v_b.contiguous())

            if prev_k is not None:
                eps = 1e-12
                for l_idx in range(n_layers):
                    for arr_curr, arr_prev, cos_out, drift_out in (
                        (curr_k[l_idx], prev_k[l_idx], cos_k, drift_k),
                        (curr_v[l_idx], prev_v[l_idx], cos_v, drift_v),
                    ):
                        a_n = arr_curr.norm(dim=-1)
                        p_n = arr_prev.norm(dim=-1)
                        dot = (arr_curr * arr_prev).sum(dim=-1)
                        cos_t = dot / (a_n * p_n + eps)
                        diff_n = (arr_curr - arr_prev).norm(dim=-1)
                        drift_t = diff_n / (a_n + eps)
                        cos_out[t, l_idx]   = cos_t.numpy()
                        drift_out[t, l_idx] = drift_t.numpy()

            prev_k, prev_v = curr_k, curr_v

            _temp = sampler.temperature[t] ** 1
            pred_code = torch.distributions.Categorical(logits=logit.float() * _temp).sample()
            code[U_t.to(cfg.device)] = pred_code.view(nb, input_size, input_size)[U_t.to(cfg.device)]

            print(f"[step {t:02d}/{T}] released {U_t.sum().item() // nb} tokens/sample (r={r})")
            prev_r = r

    return cos_k, drift_k, cos_v, drift_v, commit_step, active_at


def aggregate_and_plot(cos_k, drift_k, cos_v, drift_v, commit_step, active_at,
                       out_dir, gate, nb):
    """Compute grouped statistics, write the two figures, and return summary dict."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    T, L, B, N = cos_k.shape

    step_idx = np.arange(T)[:, None, None]
    commit_b = commit_step[None, :, :]
    in_M_t   = (commit_b >= 0) & (commit_b <= step_idx)
    inactive = ~active_at

    inactive_committed = inactive & in_M_t
    inactive_masked    = inactive & ~in_M_t

    def bcast(m):
        return np.broadcast_to(m[:, None, :, :], cos_k.shape)

    valid = ~np.isnan(cos_k)

    def grouped_mean(arr, mask, axes):
        m = mask & valid
        a_ = np.where(m, arr, 0.0)
        s = a_.sum(axis=axes)
        c = m.sum(axis=axes).clip(min=1)
        out = s / c
        out[m.sum(axis=axes) == 0] = np.nan
        return out

    def all_marginals(arr):
        return dict(
            by_layer_committed = grouped_mean(arr, bcast(inactive_committed), axes=(0, 2, 3)),
            by_layer_masked    = grouped_mean(arr, bcast(inactive_masked),    axes=(0, 2, 3)),
            by_step_committed  = grouped_mean(arr, bcast(inactive_committed), axes=(1, 2, 3)),
            by_step_masked     = grouped_mean(arr, bcast(inactive_masked),    axes=(1, 2, 3)),
            heat_committed     = grouped_mean(arr, bcast(inactive_committed), axes=(2, 3)),  # (T, L)
        )

    m_cos_k   = all_marginals(cos_k)
    m_drift_k = all_marginals(drift_k)
    m_cos_v   = all_marginals(cos_v)
    m_drift_v = all_marginals(drift_v)

    # by-age (committed only) for drift (the more readable of the two)
    age = step_idx - commit_b
    age_valid = (commit_b >= 0) & (age >= 1) & inactive
    max_age = T - 1
    drf_k_by_age = np.full(max_age + 1, np.nan, dtype=np.float32)
    drf_v_by_age = np.full(max_age + 1, np.nan, dtype=np.float32)
    for a_ in range(1, max_age + 1):
        m = bcast((age == a_) & age_valid) & valid
        if m.any():
            drf_k_by_age[a_] = drift_k[m].mean()
            drf_v_by_age[a_] = drift_v[m].mean()

    os.makedirs(out_dir, exist_ok=True)

    # ---------- 8-panel: rows = {K, V}, cols = {by layer cos, by step cos, by layer drift, by step drift} ----------
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for row, (name, m_cos, m_drift) in enumerate(
        [("K", m_cos_k, m_drift_k), ("V", m_cos_v, m_drift_v)]
    ):
        ax = axes[row, 0]
        ax.plot(m_cos["by_layer_committed"], marker="o", label="committed")
        ax.plot(m_cos["by_layer_masked"],    marker="s", label="still-masked")
        ax.axvspan(gate["lmin"], gate["lmax"], alpha=0.12, color="orange",
                   label="partial-update gate")
        ax.set_xlabel("layer"); ax.set_ylabel(f"cos({name}(t), {name}(t-1))")
        ax.set_title(f"{name}: cosine — by layer"); ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[row, 1]
        ax.plot(m_cos["by_step_committed"], marker="o", label="committed")
        ax.plot(m_cos["by_step_masked"],    marker="s", label="still-masked")
        ax.axvspan(gate["smin"], gate["smax"], alpha=0.12, color="orange",
                   label="partial-update gate")
        ax.set_xlabel("step (t)"); ax.set_ylabel("cos")
        ax.set_title(f"{name}: cosine — by step"); ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[row, 2]
        ax.plot(m_drift["by_layer_committed"], marker="o", label="committed")
        ax.plot(m_drift["by_layer_masked"],    marker="s", label="still-masked")
        ax.axvspan(gate["lmin"], gate["lmax"], alpha=0.12, color="orange",
                   label="partial-update gate")
        ax.set_xlabel("layer"); ax.set_ylabel("rel drift")
        ax.set_title(f"{name}: rel drift — by layer"); ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[row, 3]
        ax.plot(m_drift["by_step_committed"], marker="o", label="committed")
        ax.plot(m_drift["by_step_masked"],    marker="s", label="still-masked")
        ax.axvspan(gate["smin"], gate["smax"], alpha=0.12, color="orange",
                   label="partial-update gate")
        ax.set_xlabel("step (t)"); ax.set_ylabel("rel drift")
        ax.set_title(f"{name}: rel drift — by step"); ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p_main = os.path.join(out_dir, "kv_drift.png")
    plt.savefig(p_main, dpi=120); plt.close(fig)
    print("[plot] saved", p_main)

    # ---------- (layer × step) heatmaps for K and V, committed-inactive ----------
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for row, (name, heat_cos, heat_drift) in enumerate(
        [("K", m_cos_k["heat_committed"], m_drift_k["heat_committed"]),
         ("V", m_cos_v["heat_committed"], m_drift_v["heat_committed"])]
    ):
        ax = axes[row, 0]
        im = ax.imshow(heat_cos.T, aspect="auto", origin="lower", cmap="viridis")
        rect = Rectangle((gate["smin"] - 0.5, gate["lmin"] - 0.5),
                         gate["smax"] - gate["smin"] + 1,
                         gate["lmax"] - gate["lmin"] + 1,
                         linewidth=1.5, edgecolor="white", facecolor="none",
                         linestyle="--")
        ax.add_patch(rect)
        ax.set_xlabel("step (t)"); ax.set_ylabel("layer")
        ax.set_title(f"{name}: mean cosine — committed inactive (dashed = partial-update gate)")
        fig.colorbar(im, ax=ax)

        ax = axes[row, 1]
        im = ax.imshow(heat_drift.T, aspect="auto", origin="lower", cmap="magma")
        rect = Rectangle((gate["smin"] - 0.5, gate["lmin"] - 0.5),
                         gate["smax"] - gate["smin"] + 1,
                         gate["lmax"] - gate["lmin"] + 1,
                         linewidth=1.5, edgecolor="white", facecolor="none",
                         linestyle="--")
        ax.add_patch(rect)
        ax.set_xlabel("step (t)"); ax.set_ylabel("layer")
        ax.set_title(f"{name}: mean rel-drift — committed inactive")
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    p_heat = os.path.join(out_dir, "kv_heatmap.png")
    plt.savefig(p_heat, dpi=120); plt.close(fig)
    print("[plot] saved", p_heat)

    # ---------- text summary, focused on the partial-update gate ----------
    smin, smax, lmin, lmax = gate["smin"], gate["smax"], gate["lmin"], gate["lmax"]
    gate_heat_k = m_drift_k["heat_committed"][smin:smax + 1, lmin:lmax + 1]
    gate_heat_v = m_drift_v["heat_committed"][smin:smax + 1, lmin:lmax + 1]
    gate_cos_k  = m_cos_k  ["heat_committed"][smin:smax + 1, lmin:lmax + 1]
    gate_cos_v  = m_cos_v  ["heat_committed"][smin:smax + 1, lmin:lmax + 1]

    print("\n=== Summary (committed-inactive tokens) ===")
    print(f"layers: {L}, steps: {T}, samples: {B}, tokens/sample: {N}")
    print(f"Layer-0 K drift (should be ~0 if proj=1, cond is step-invariant): "
          f"{np.nanmean(m_drift_k['heat_committed'][:, 0]):.6f}")
    print(f"Layer-0 V drift: "
          f"{np.nanmean(m_drift_v['heat_committed'][:, 0]):.6f}")
    print(f"K drift in partial gate (l={lmin}..{lmax}, t={smin}..{smax}): "
          f"mean={np.nanmean(gate_heat_k):.4f} max={np.nanmax(gate_heat_k):.4f}")
    print(f"V drift in partial gate: "
          f"mean={np.nanmean(gate_heat_v):.4f} max={np.nanmax(gate_heat_v):.4f}")
    print(f"K cos in partial gate:  mean={np.nanmean(gate_cos_k):.4f} "
          f"min={np.nanmin(gate_cos_k):.4f}")
    print(f"V cos in partial gate:  mean={np.nanmean(gate_cos_v):.4f} "
          f"min={np.nanmin(gate_cos_v):.4f}")

    # ---------- cond vs uncond split (sanity check; both halves go through cache in real inference) ----------
    # Slice raw drift arrays on the B axis: [:nb] = cond, [nb:] = uncond.
    def gate_mean_drift_split(arr, b_slice):
        # arr: (T, L, B, N) -> restrict to gate steps/layers + committed mask + b_slice
        sub = arr[smin:smax + 1, lmin:lmax + 1, b_slice]      # (gT, gL, nb, N)
        m_active = active_at[smin:smax + 1, b_slice]          # (gT, nb, N)
        m_commit_b = commit_step[None, b_slice, :] >= 0       # (1, nb, N)
        # in_M_t for the gate steps
        step_axis = np.arange(smin, smax + 1)[:, None, None]  # (gT, 1, 1)
        m_in_M_t = (commit_step[None, b_slice, :] <= step_axis) & m_commit_b
        m_committed_inactive = (~m_active) & m_in_M_t         # (gT, nb, N)
        m_full = np.broadcast_to(m_committed_inactive[:, None, :, :], sub.shape)
        valid_sub = ~np.isnan(sub)
        m_all = m_full & valid_sub
        if not m_all.any():
            return float("nan")
        return float(sub[m_all].mean())

    print("\n=== cond vs uncond consistency (gate region only) ===")
    for name, arr in (("K drift", drift_k), ("V drift", drift_v)):
        c = gate_mean_drift_split(arr, slice(0, nb))
        u = gate_mean_drift_split(arr, slice(nb, 2 * nb))
        ratio = (u / c) if (c and not np.isnan(c)) else float("nan")
        print(f"  {name}: cond={c:.4f}  uncond={u:.4f}  uncond/cond={ratio:.3f}")
    print("  (ratios near 1.0 ⇒ same drift behavior on both halves; "
          "large gaps ⇒ cache may need per-half tuning)")

    return dict(
        cos_k_by_layer=m_cos_k["by_layer_committed"],
        cos_v_by_layer=m_cos_v["by_layer_committed"],
        drift_k_by_layer=m_drift_k["by_layer_committed"],
        drift_v_by_layer=m_drift_v["by_layer_committed"],
        cos_k_by_step=m_cos_k["by_step_committed"],
        cos_v_by_step=m_cos_v["by_step_committed"],
        drift_k_by_step=m_drift_k["by_step_committed"],
        drift_v_by_step=m_drift_v["by_step_committed"],
        drift_k_by_age=drf_k_by_age,
        drift_v_by_age=drf_v_by_age,
        cos_k_layer_step=m_cos_k["heat_committed"],
        cos_v_layer_step=m_cos_v["heat_committed"],
        drift_k_layer_step=m_drift_k["heat_committed"],
        drift_v_layer_step=m_drift_v["heat_committed"],
    )


def main():
    args_cli = parse_args()
    gate = dict(
        lmin=args_cli.gate_layer_min, lmax=args_cli.gate_layer_max,
        smin=args_cli.gate_step_min,  smax=args_cli.gate_step_max,
    )

    cos_k, drift_k, cos_v, drift_v, commit_step, active_at = run_inference_capture(args_cli)
    print(f"[capture] K/V cos/drift shape = {cos_k.shape}")

    summary = aggregate_and_plot(
        cos_k, drift_k, cos_v, drift_v, commit_step, active_at,
        args_cli.out_dir, gate, nb=args_cli.nb_sample,
    )

    out_npz = os.path.join(args_cli.out_dir, "kv_drift.npz")
    np.savez_compressed(
        out_npz,
        cos_k=cos_k.astype(np.float16), drift_k=drift_k.astype(np.float16),
        cos_v=cos_v.astype(np.float16), drift_v=drift_v.astype(np.float16),
        commit_step=commit_step, active_at=active_at,
        **summary,
    )
    print("[save] raw + summary stats:", out_npz)


if __name__ == "__main__":
    main()
