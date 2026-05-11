"""
Quantify FFN-delta stability across Halton-MaskGIT decoding steps.

For vanilla (full-token-update) inference, at every step t and every transformer
layer l we record ffn_delta_l(t) per token, then for each token that is
*inactive* at step t (i.e. NOT newly released by the Halton schedule at this
step) we compute:

    cos_l(t)   = cosine( ffn_delta_l(t), ffn_delta_l(t-1) )
    drift_l(t) = || ffn_delta_l(t) - ffn_delta_l(t-1) ||_2 / || ffn_delta_l(t) ||_2

Stats are then grouped by:
  - layer depth
  - step progress
  - token age (steps since the token was first committed; only meaningful for
    already-committed tokens)

Inactive tokens fall into two distinct sub-populations that we plot
separately, since they behave very differently:
  (a) committed-before-t: token's input id is fixed, only context drifts
  (b) still-masked:       token's input is mask_value, both at t and t-1

Outputs:
  statics/ffn_delta/ffn_delta_stability.npz   raw per-token cos / drift arrays
  statics/ffn_delta/ffn_delta_stability.png   6-panel grouped plot
  statics/ffn_delta/ffn_delta_heatmap.png     layer x step heatmaps
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
from Network.transformer import Block, modulate


_DELTA_BUFFER: list = []


def _patched_block_forward(self, x, cond, mask=None, active_mask=None):
    """Mirror of Block.forward that also pushes the FFN delta into _DELTA_BUFFER."""
    gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.mlp(cond).chunk(6, dim=1)
    x = x + alpha1.unsqueeze(1) * self.attn(
        modulate(self.ln1(x), gamma1, beta1),
        mask=mask,
        active_mask=active_mask,
    )
    if active_mask is None:
        ff_out = self.ff(modulate(self.ln2(x), gamma2, beta2))
        delta = alpha2.unsqueeze(1) * ff_out
    else:
        b, h_w, d = x.shape
        n_active = int(active_mask[0].sum().item())
        x_active = x[active_mask].view(b, n_active, d)
        ff_out = self.ff(modulate(self.ln2(x_active), gamma2, beta2))
        delta = torch.zeros_like(x)
        delta[active_mask] = (alpha2.unsqueeze(1) * ff_out).reshape(b * n_active, d)
    _DELTA_BUFFER.append(delta.detach().to(torch.float32).cpu())
    return x + delta


def patch_blocks(transformer):
    for blk in transformer.layers:
        blk.forward = _patched_block_forward.__get__(blk, Block)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     type=str, default="Config/base_cls2img.yaml")
    p.add_argument("--vit-size",   type=str, default="large")
    p.add_argument("--img-size",   type=int, default=384)
    p.add_argument("--steps",      type=int, default=32)
    p.add_argument("--nb-sample",  type=int, default=4)
    p.add_argument("--cfg-w",      type=float, default=2.0)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--dtype",      type=str, default="bfloat16",
                   help="bfloat16|float32 — bf16 is much faster, deltas are cast to fp32 for math")
    p.add_argument("--out-dir",    type=str, default="statics/ffn_delta")
    return p.parse_args()


def run_inference_capture(args_cli):
    """Run vanilla 32-step Halton sampling and stream cos/drift step by step.
    Returns:
        cos:         (steps, layers, b, n_tokens) float32 — cosine vs prev step (NaN at step 0)
        drift:       (steps, layers, b, n_tokens) float32 — rel drift vs prev step
        commit_step: (b, n_tokens) int — step at which each token was first released (-1 if never)
        active_at:   (steps, b, n_tokens) bool — was this token in U_t at step t
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

    patch_blocks(transformer.transformer)
    n_layers = len(transformer.transformer.layers)
    register = transformer.register
    print(f"[init] patched {n_layers} blocks, register={register}")

    sampler = HaltonSampler(
        sm_temp_min=1, sm_temp_max=1.2, temp_pow=1, temp_warmup=0,
        w=args_cli.cfg_w, sched_pow=2, step=args_cli.steps,
        randomize=False, top_k=-1,
    )

    input_size = trainer.input_size  # e.g. 24 for 384/16
    nb = args_cli.nb_sample
    n_tokens = input_size * input_size

    # Pre-compute U_t / M_t schedule (matches the loop below since randomize=False).
    l_U_t, _ = sampler.compute_schedule(input_size, nb_sample=nb)
    active_at = torch.stack(l_U_t).view(args_cli.steps, nb, n_tokens).bool().numpy()

    # commit_step: when each token is first released
    commit_step = -np.ones((nb, n_tokens), dtype=np.int64)
    for t in range(args_cli.steps):
        new = active_at[t] & (commit_step == -1)
        commit_step[new] = t

    # Default-class labels (first nb_sample of the demo classes).
    demo_labels = [1, 7, 282, 604, 724, 179, 681, 850, 850]
    labels = torch.LongTensor(demo_labels[:nb]).to(cfg.device)

    trainer.vit.eval()
    drop = torch.ones(nb, dtype=torch.bool, device=cfg.device)
    code = torch.full((nb, input_size, input_size), cfg.mask_value,
                      dtype=torch.long, device=cfg.device)

    halton_mask = sampler.basic_halton_mask.clone().unsqueeze(0).expand(nb, n_tokens, 2)

    T = args_cli.steps
    cos   = np.full((T, n_layers, nb, n_tokens), np.nan, dtype=np.float32)
    drift = np.full((T, n_layers, nb, n_tokens), np.nan, dtype=np.float32)

    prev_deltas = None  # list[Tensor] length n_layers, each (nb, n_tokens, d) on CPU fp32

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

            _DELTA_BUFFER.clear()
            with trainer.autocast:
                logit = trainer.vit(
                    torch.cat([code.clone(), code.clone()], dim=0),
                    torch.cat([labels, labels], dim=0),
                    torch.cat([~drop, drop], dim=0),
                    active_mask=None,
                )
            logit_c, logit_u = torch.chunk(logit, 2, dim=0)
            logit = (1 + sampler.w) * logit_c - sampler.w * logit_u

            assert len(_DELTA_BUFFER) == n_layers, f"got {len(_DELTA_BUFFER)} deltas, expected {n_layers}"
            curr_deltas = []
            for d in _DELTA_BUFFER:
                d_cond = d[:nb]
                if register > 0:
                    d_cond = d_cond[:, :-register]
                curr_deltas.append(d_cond.contiguous())  # (nb, n_tokens, d)

            # Pairwise cos / drift vs previous step (per token, per layer).
            if prev_deltas is not None:
                eps = 1e-12
                for l_idx in range(n_layers):
                    a = curr_deltas[l_idx]
                    p = prev_deltas[l_idx]
                    a_n = a.norm(dim=-1)
                    p_n = p.norm(dim=-1)
                    dot = (a * p).sum(dim=-1)
                    cos_t = dot / (a_n * p_n + eps)
                    diff_n = (a - p).norm(dim=-1)
                    drift_t = diff_n / (a_n + eps)
                    cos[t, l_idx]   = cos_t.numpy()
                    drift[t, l_idx] = drift_t.numpy()

            prev_deltas = curr_deltas  # next step's "previous"

            _temp = sampler.temperature[t] ** 1
            prob = torch.softmax((logit.float() * _temp), dim=-1)
            pred_code = torch.distributions.Categorical(probs=prob).sample()
            code[U_t.to(cfg.device)] = pred_code.view(nb, input_size, input_size)[U_t.to(cfg.device)]

            print(f"[step {t:02d}/{T}] released {U_t.sum().item() // nb} tokens/sample (cumulative r={r})")
            prev_r = r

    return cos, drift, commit_step, active_at


def aggregate_and_plot(cos, drift, commit_step, active_at, out_dir):
    """Compute grouped statistics and write plots + summary table."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T, L, B, N = cos.shape

    # in_M_t[t, b, n]: token already released by step t (i.e. commit_step in [0, t]).
    step_idx = np.arange(T)[:, None, None]            # (T, 1, 1)
    commit_b = commit_step[None, :, :]                # (1, B, N)
    in_M_t = (commit_b >= 0) & (commit_b <= step_idx)  # (T, B, N)
    inactive = ~active_at                              # (T, B, N)

    inactive_committed = inactive & in_M_t            # already-committed inactive
    inactive_masked    = inactive & ~in_M_t           # still-masked inactive

    # Broadcast (T, B, N) masks over the L axis to match (T, L, B, N) arrays.
    def bcast(m):
        return np.broadcast_to(m[:, None, :, :], cos.shape)

    valid = ~np.isnan(cos)

    def grouped_mean(arr, mask, axes):
        m = mask & valid
        a_ = np.where(m, arr, 0.0)
        s = a_.sum(axis=axes)
        c = m.sum(axis=axes).clip(min=1)
        out = s / c
        # Re-mark groups with zero count as NaN
        out[m.sum(axis=axes) == 0] = np.nan
        return out

    # ----- by layer (collapse over T, B, N) -----
    cos_l_c = grouped_mean(cos,   bcast(inactive_committed), axes=(0, 2, 3))
    cos_l_m = grouped_mean(cos,   bcast(inactive_masked),    axes=(0, 2, 3))
    drf_l_c = grouped_mean(drift, bcast(inactive_committed), axes=(0, 2, 3))
    drf_l_m = grouped_mean(drift, bcast(inactive_masked),    axes=(0, 2, 3))

    # ----- by step (collapse over L, B, N) -----
    cos_s_c = grouped_mean(cos,   bcast(inactive_committed), axes=(1, 2, 3))
    cos_s_m = grouped_mean(cos,   bcast(inactive_masked),    axes=(1, 2, 3))
    drf_s_c = grouped_mean(drift, bcast(inactive_committed), axes=(1, 2, 3))
    drf_s_m = grouped_mean(drift, bcast(inactive_masked),    axes=(1, 2, 3))

    # ----- by token age (committed only). age = t - commit_step >= 1 -----
    age = step_idx - commit_b                  # (T, B, N) — undefined where commit==-1
    age_valid = (commit_b >= 0) & (age >= 1) & inactive  # ⇔ inactive & in_M_t
    max_age = T - 1
    cos_by_age = np.full(max_age + 1, np.nan, dtype=np.float32)
    drf_by_age = np.full(max_age + 1, np.nan, dtype=np.float32)
    for a_ in range(1, max_age + 1):
        m = bcast((age == a_) & age_valid) & valid
        if m.any():
            cos_by_age[a_] = cos[m].mean()
            drf_by_age[a_] = drift[m].mean()

    # ----- (layer, step) heatmap of relative drift, committed group -----
    heat_drift = grouped_mean(drift, bcast(inactive_committed), axes=(2, 3))   # (T, L)
    heat_cos   = grouped_mean(cos,   bcast(inactive_committed), axes=(2, 3))   # (T, L)

    os.makedirs(out_dir, exist_ok=True)

    # ---------- 6-panel summary ----------
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    ax = axes[0, 0]
    ax.plot(cos_l_c, marker="o", label="committed inactive")
    ax.plot(cos_l_m, marker="s", label="still-masked inactive")
    ax.set_xlabel("layer index"); ax.set_ylabel("cos(δ_l(t), δ_l(t-1))")
    ax.set_title("cosine vs prev step — by layer")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(cos_s_c, marker="o", label="committed inactive")
    ax.plot(cos_s_m, marker="s", label="still-masked inactive")
    ax.set_xlabel("step (t)"); ax.set_ylabel("cos")
    ax.set_title("cosine vs prev step — by step")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ages = np.arange(len(cos_by_age))
    ax.plot(ages, cos_by_age, marker="o")
    ax.set_xlabel("token age (steps since first commit)"); ax.set_ylabel("cos")
    ax.set_title("cosine — by token age (committed)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(drf_l_c, marker="o", label="committed inactive")
    ax.plot(drf_l_m, marker="s", label="still-masked inactive")
    ax.set_xlabel("layer index"); ax.set_ylabel("‖Δδ‖₂ / ‖δ(t)‖₂")
    ax.set_title("relative drift — by layer")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(drf_s_c, marker="o", label="committed inactive")
    ax.plot(drf_s_m, marker="s", label="still-masked inactive")
    ax.set_xlabel("step (t)"); ax.set_ylabel("rel drift")
    ax.set_title("relative drift — by step")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.plot(ages, drf_by_age, marker="o")
    ax.set_xlabel("token age"); ax.set_ylabel("rel drift")
    ax.set_title("relative drift — by token age (committed)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p_main = os.path.join(out_dir, "ffn_delta_stability.png")
    plt.savefig(p_main, dpi=120); plt.close(fig)
    print("[plot] saved", p_main)

    # ---------- (layer × step) heatmaps for committed-inactive ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    im = axes[0].imshow(heat_cos.T, aspect="auto", origin="lower", cmap="viridis")
    axes[0].set_xlabel("step (t)"); axes[0].set_ylabel("layer")
    axes[0].set_title("mean cosine — committed inactive tokens")
    fig.colorbar(im, ax=axes[0])

    im = axes[1].imshow(heat_drift.T, aspect="auto", origin="lower", cmap="magma")
    axes[1].set_xlabel("step (t)"); axes[1].set_ylabel("layer")
    axes[1].set_title("mean rel-drift — committed inactive tokens")
    fig.colorbar(im, ax=axes[1])

    plt.tight_layout()
    p_heat = os.path.join(out_dir, "ffn_delta_heatmap.png")
    plt.savefig(p_heat, dpi=120); plt.close(fig)
    print("[plot] saved", p_heat)

    # ---------- print quick text summary ----------
    print("\n=== Summary (committed-inactive tokens) ===")
    print(f"layers: {L}, steps: {T}, samples: {B}, tokens/sample: {N}")
    print(f"cos: layer mean range  [{np.nanmin(cos_l_c):.4f}, {np.nanmax(cos_l_c):.4f}]")
    print(f"cos: step  mean range  [{np.nanmin(cos_s_c):.4f}, {np.nanmax(cos_s_c):.4f}]")
    print(f"drift: layer mean range[{np.nanmin(drf_l_c):.4f}, {np.nanmax(drf_l_c):.4f}]")
    print(f"drift: step  mean range[{np.nanmin(drf_s_c):.4f}, {np.nanmax(drf_s_c):.4f}]")
    if np.any(~np.isnan(cos_by_age)):
        a_min, a_max = np.nanargmin(cos_by_age), np.nanargmax(cos_by_age)
        print(f"cos by age — min={cos_by_age[a_min]:.4f} @ age={a_min}, "
              f"max={cos_by_age[a_max]:.4f} @ age={a_max}")

    return dict(
        cos_by_layer_committed=cos_l_c, cos_by_layer_masked=cos_l_m,
        cos_by_step_committed=cos_s_c,  cos_by_step_masked=cos_s_m,
        drift_by_layer_committed=drf_l_c, drift_by_layer_masked=drf_l_m,
        drift_by_step_committed=drf_s_c,  drift_by_step_masked=drf_s_m,
        cos_by_age=cos_by_age, drift_by_age=drf_by_age,
        cos_layer_step=heat_cos, drift_layer_step=heat_drift,
    )


def main():
    args_cli = parse_args()

    cos, drift, commit_step, active_at = run_inference_capture(args_cli)
    print(f"[capture] cos/drift shape = {cos.shape}")

    summary = aggregate_and_plot(cos, drift, commit_step, active_at, args_cli.out_dir)

    out_npz = os.path.join(args_cli.out_dir, "ffn_delta_stability.npz")
    np.savez_compressed(
        out_npz,
        cos=cos.astype(np.float16),       # halve disk size; analysis already done
        drift=drift.astype(np.float16),
        commit_step=commit_step,
        active_at=active_at,
        **summary,
    )
    print("[save] raw + summary stats:", out_npz)


if __name__ == "__main__":
    main()
