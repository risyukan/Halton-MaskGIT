"""
KV-cache substitution ablation: run end-to-end sampling under several "cache
region" hypotheses and compare to vanilla.

Pure runtime monkey-patch — does NOT modify Network/transformer.py,
Sampler/halton_sampler.py, or the existing FFN-cache logic. The patched
Attention.forward replaces K and/or V at committed-age>=2 positions with the
previous step's K/V (mirroring the FFN-cache "running" semantics: errors do
compound across steps, same as the existing self.cached_ffn_delta scheme).

Configs (A/B/C/D + vanilla baseline):
  none: no cache (this is the ground truth used for comparison)
  A: full partial-update gate (layer 3-21, step 5-30), K and V cached
  B: shallow only (layer 0-10, step 5-30), K and V cached  — test the
     hypothesis that mid-deep layers (10-22) carry most drift
  C: late steps only (layer 3-21, step 15-30), K and V cached  — test the
     hypothesis that early-gate (step 5-10) carries most drift
  D: K-only, full gate (V re-computed) — V drift was ~50% larger than K,
     so this asks: is V the bottleneck?

Each config runs independently with the same seed and same labels, so divergence
from vanilla comes from cache approximation alone (modulo CUDA non-determinism).

Outputs (statics/kv_substitution_ablation/):
  grid_<config>.png          — 8-image grid for visual inspection
  comparison_grid.png        — side-by-side {vanilla, A, B, C, D}
  summary.txt                — per-config code-disagreement + pixel MSE vs vanilla
  codes_<config>.pt          — saved final codes for downstream FID if wanted
  images_<config>.pt         — saved final images
"""

import argparse
import math
import os
import sys
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Utils.utils import load_args_from_file
from Trainer.cls_trainer import MaskGIT
from Sampler.halton_sampler import HaltonSampler
from Network.transformer import Attention


# ──────────────────────────────────────────────────────────────────────────────
# Ablation controller — global state shared with the patched Attention.forward
# ──────────────────────────────────────────────────────────────────────────────

class AblationController:
    """Holds per-config state for one full sampling run.

    apply_k(l, t), apply_v(l, t): policy — should this (layer, step) pair use
    cached K (resp. V) for committed-age>=2 positions?
    """

    def __init__(self, name: str, n_layers: int,
                 apply_k: Callable[[int, int], bool],
                 apply_v: Callable[[int, int], bool]):
        self.name = name
        self.n_layers = n_layers
        self.apply_k = apply_k
        self.apply_v = apply_v

        # Set per-step before each step's forward.
        self.cur_step: int = 0
        # (2*nb, seq_len) bool — committed-age>=2 mask aligned with full sequence
        # (content tokens + register tail set to False). Registered each step.
        self.cur_mask: Optional[torch.Tensor] = None

        # Running cache: previous-step (possibly substituted) K and V per layer.
        # Each is (2*nb, seq_len, d_model) fp16/fp32 on device.
        self.prev_k: List[Optional[torch.Tensor]] = [None] * n_layers
        self.prev_v: List[Optional[torch.Tensor]] = [None] * n_layers

    def reset_cache(self):
        self.prev_k = [None] * self.n_layers
        self.prev_v = [None] * self.n_layers
        self.cur_step = 0
        self.cur_mask = None


CTRL: Optional[AblationController] = None


# ──────────────────────────────────────────────────────────────────────────────
# Patched Attention.forward — mirrors the full-update branch, plus substitution
# ──────────────────────────────────────────────────────────────────────────────

def _patched_attn_forward(self, x, mask=None, active_idx=None):
    """Vanilla forward with optional K/V substitution at committed-age>=2 positions.

    Substitution is gated by CTRL.apply_k / CTRL.apply_v at (layer, step). When
    triggered, the current K (resp. V) at committed-age>=2 positions is
    overwritten with the previous step's K (resp. V) at the SAME positions.
    Then current (possibly substituted) K/V is stored back for the next step —
    this is the "running" cache semantics, errors propagate.

    active_idx must be None: this analyzer only runs the full-update path.
    """
    assert active_idx is None, "substitution ablation uses full-update forward"
    assert CTRL is not None, "AblationController not set"
    l = self._layer_idx
    t = CTRL.cur_step

    b, h_w, _ = x.shape
    xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
    xq, xk = self.qk_norm(xq, xk, xv)

    do_k = CTRL.apply_k(l, t) and (CTRL.prev_k[l] is not None) and (CTRL.cur_mask is not None)
    do_v = CTRL.apply_v(l, t) and (CTRL.prev_v[l] is not None) and (CTRL.cur_mask is not None)

    if do_k or do_v:
        m = CTRL.cur_mask  # (b, h_w) bool on device
        m3 = m.unsqueeze(-1)  # broadcast over feature dim
        if do_k:
            xk = torch.where(m3, CTRL.prev_k[l], xk)
        if do_v:
            xv = torch.where(m3, CTRL.prev_v[l], xv)

    # Save current K/V for next step (running cache).
    CTRL.prev_k[l] = xk.detach()
    CTRL.prev_v[l] = xv.detach()

    # ── standard attention from here, mirroring transformer.py:80-111 ──
    xq = xq.view(b, h_w, self.n_local_heads, self.head_dim)
    xk = xk.view(b, h_w, self.n_local_heads, self.head_dim)
    xv = xv.view(b, h_w, self.n_local_heads, self.head_dim)
    xq, xk, xv = (t_.transpose(1, 2) for t_ in (xq, xk, xv))

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
    return self.wo(output)


def patch_attention(transformer):
    """Rebind Attention.forward on every block and tag layer indices."""
    for i, blk in enumerate(transformer.layers):
        blk.attn._layer_idx = i
        blk.attn.forward = _patched_attn_forward.__get__(blk.attn, Attention)


# ──────────────────────────────────────────────────────────────────────────────
# Config registry
# ──────────────────────────────────────────────────────────────────────────────

def make_configs(gate_lmin=3, gate_lmax=21, gate_smin=5, gate_smax=30):
    """Build the standard A/B/C/D configs around the existing partial-update gate."""

    def in_full_gate(l, t):
        return gate_lmin <= l <= gate_lmax and gate_smin <= t <= gate_smax

    def in_shallow_gate(l, t):
        return 0 <= l <= 10 and gate_smin <= t <= gate_smax

    def in_late_gate(l, t):
        return gate_lmin <= l <= gate_lmax and 15 <= t <= gate_smax

    return {
        "none": dict(apply_k=lambda l, t: False, apply_v=lambda l, t: False),
        "A":    dict(apply_k=in_full_gate,        apply_v=in_full_gate),
        "B":    dict(apply_k=in_shallow_gate,     apply_v=in_shallow_gate),
        "C":    dict(apply_k=in_late_gate,        apply_v=in_late_gate),
        "D":    dict(apply_k=in_full_gate,        apply_v=lambda l, t: False),
    }


# ──────────────────────────────────────────────────────────────────────────────
# One sampling run
# ──────────────────────────────────────────────────────────────────────────────

def run_one_config(trainer, sampler, cfg, name, apply_k, apply_v, labels, seed):
    """Run a full 32-step sampling under one ablation config. Returns (images, codes)."""
    global CTRL
    nb = labels.shape[0]
    register = trainer.vit.register
    input_size = trainer.input_size
    n_tokens = input_size * input_size
    n_layers = len(trainer.vit.transformer.layers)
    T = sampler.step

    # Pre-compute committed-age>=2 mask aligned with the model's full sequence
    # (content + register tail). cond and uncond halves see the same Halton
    # schedule, so we tile across the 2*nb effective batch.
    l_U_t, _ = sampler.compute_schedule(input_size, nb_sample=nb)
    active_at = torch.stack(l_U_t).view(T, nb, n_tokens).bool()  # (T, nb, N)
    commit_step = torch.full((nb, n_tokens), -1, dtype=torch.long)
    for t in range(T):
        new = active_at[t] & (commit_step == -1)
        commit_step[new] = t
    step_idx = torch.arange(T)[:, None, None]
    commit_b = commit_step[None, :, :]
    age = step_idx - commit_b
    not_active = ~active_at
    committed_age2 = (commit_b >= 0) & (age >= 2) & not_active  # (T, nb, N)
    # Tile cond+uncond, then pad with register tail of False.
    committed_age2 = committed_age2.repeat(1, 2, 1)              # (T, 2*nb, N)
    if register > 0:
        pad = torch.zeros(T, 2 * nb, register, dtype=torch.bool)
        committed_age2 = torch.cat([committed_age2, pad], dim=2)
    # Move to device once.
    committed_age2 = committed_age2.to(cfg.device)

    # Fresh controller per config.
    CTRL = AblationController(name, n_layers, apply_k, apply_v)

    # Reset seed before each run for reproducibility.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    trainer.vit.eval()
    drop = torch.ones(nb, dtype=torch.bool, device=cfg.device)
    code = torch.full((nb, input_size, input_size), cfg.mask_value,
                      dtype=torch.long, device=cfg.device)

    halton_mask = sampler.basic_halton_mask.clone().unsqueeze(0).expand(nb, n_tokens, 2)

    with torch.no_grad():
        prev_r = 0
        for t in range(T):
            CTRL.cur_step = t
            CTRL.cur_mask = committed_age2[t]

            ratio = (t + 1) / T
            r = 1 - (torch.arccos(torch.tensor(ratio)) / (math.pi * 0.5))
            r = int(r * (input_size ** 2))
            r = max(t + 1, r)

            _u = halton_mask[:, prev_r:r]
            U_t = torch.zeros(nb, input_size, input_size, dtype=torch.bool)
            for i in range(nb):
                U_t[i, _u[i, :, 0], _u[i, :, 1]] = True

            with trainer.autocast:
                logit = trainer.vit(
                    torch.cat([code.clone(), code.clone()], dim=0),
                    torch.cat([labels, labels], dim=0),
                    torch.cat([~drop, drop], dim=0),
                    active_mask=None,
                )
            logit_c, logit_u = torch.chunk(logit, 2, dim=0)
            logit = (1 + sampler.w) * logit_c - sampler.w * logit_u

            _temp = sampler.temperature[t] ** 1
            pred_code = torch.distributions.Categorical(
                logits=logit.float() * _temp
            ).sample()
            code[U_t.to(cfg.device)] = pred_code.view(
                nb, input_size, input_size
            )[U_t.to(cfg.device)]
            prev_r = r

        code = torch.clamp(code, 0, cfg.codebook_size - 1)
        images = trainer.ae.decode_code(code)
        images = torch.clamp(images, -1, 1)

    return images.cpu(), code.cpu()


# ──────────────────────────────────────────────────────────────────────────────
# Eval / plotting
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(images_v, codes_v, images_a, codes_a):
    """Compare ablation result to vanilla. Returns dict of summary stats."""
    pixel_mse = float(((images_v - images_a) ** 2).mean())
    pixel_l1  = float((images_v - images_a).abs().mean())
    # vqgan-code disagreement: fraction of positions with different code
    code_disagree = float((codes_v != codes_a).float().mean())
    # per-image pixel MSE for distribution
    per_img_mse = ((images_v - images_a) ** 2).mean(dim=(1, 2, 3))
    return dict(
        pixel_mse=pixel_mse, pixel_l1=pixel_l1,
        code_disagree=code_disagree,
        per_img_mse_min=float(per_img_mse.min()),
        per_img_mse_max=float(per_img_mse.max()),
        per_img_mse_median=float(per_img_mse.median()),
    )


def save_grid(images, path, ncol=4):
    """Save image batch as a grid PNG. images in [-1, 1]."""
    import torchvision.utils as vutils
    imgs01 = (images.clamp(-1, 1) + 1) / 2.0
    grid = vutils.make_grid(imgs01, nrow=ncol, padding=2)
    vutils.save_image(grid, path)


def save_comparison_grid(images_per_config: Dict[str, torch.Tensor], path, ncol_per_row=8):
    """Stack {vanilla, A, B, C, D} vertically: each row is one config."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torchvision.utils as vutils

    names = list(images_per_config.keys())
    n_rows = len(names)
    rows = []
    for n in names:
        imgs = images_per_config[n][:ncol_per_row]
        imgs01 = (imgs.clamp(-1, 1) + 1) / 2.0
        grid_row = vutils.make_grid(imgs01, nrow=ncol_per_row, padding=2)
        rows.append(grid_row)

    fig, axes = plt.subplots(n_rows, 1, figsize=(2.5 * ncol_per_row, 2.5 * n_rows))
    if n_rows == 1:
        axes = [axes]
    for ax, name, row in zip(axes, names, rows):
        ax.imshow(row.permute(1, 2, 0).numpy())
        ax.set_title(f"config: {name}", fontsize=10, loc="left")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     type=str, default="Config/base_cls2img.yaml")
    p.add_argument("--vit-size",   type=str, default="large")
    p.add_argument("--img-size",   type=int, default=384)
    p.add_argument("--steps",      type=int, default=32)
    p.add_argument("--nb-sample",  type=int, default=8,
                   help="number of images per config (kept small for runtime; "
                        "use launch/eval_fid_* for real FID).")
    p.add_argument("--cfg-w",      type=float, default=0.5)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--dtype",      type=str, default="float32")
    p.add_argument("--out-dir",    type=str,
                   default="statics/kv_substitution_ablation")
    p.add_argument("--configs",    type=str, default="none,A,B,C,D",
                   help="comma-separated config names to run (none = vanilla)")
    p.add_argument("--gate-layer-min", type=int, default=3)
    p.add_argument("--gate-layer-max", type=int, default=21)
    p.add_argument("--gate-step-min",  type=int, default=5)
    p.add_argument("--gate-step-max",  type=int, default=30)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = load_args_from_file(args.config)
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.vit_size = args.vit_size
    cfg.img_size = args.img_size
    cfg.compile  = False
    cfg.dtype    = args.dtype
    cfg.resume   = True
    cfg.vit_folder = f"./saved_networks/ImageNet_{cfg.img_size}_{cfg.vit_size}.pth"
    cfg.data_folder = ""
    cfg.eval_folder = ""
    cfg.writer_log  = ""
    cfg.debug = True

    if args.seed >= 0:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)

    print(f"[init] loading MaskGIT (vit_size={cfg.vit_size}, img_size={cfg.img_size}, dtype={cfg.dtype})")
    trainer = MaskGIT(cfg)
    transformer = trainer.vit if not hasattr(trainer.vit, "module") else trainer.vit.module
    patch_attention(transformer.transformer)
    n_layers = len(transformer.transformer.layers)
    print(f"[init] patched {n_layers} attention modules")

    sampler = HaltonSampler(
        sm_temp_min=1, sm_temp_max=1.0, temp_pow=1, temp_warmup=1,
        w=args.cfg_w, sched_pow=2, step=args.steps, randomize=False, top_k=-1,
    )

    # Labels: same set across all configs for like-for-like comparison.
    demo_labels = [1, 7, 282, 604, 724, 179, 751, 404, 850]
    labels = torch.LongTensor(demo_labels[:args.nb_sample]).to(cfg.device)

    configs_all = make_configs(
        args.gate_layer_min, args.gate_layer_max,
        args.gate_step_min,  args.gate_step_max,
    )
    requested = [c.strip() for c in args.configs.split(",") if c.strip()]
    for c in requested:
        if c not in configs_all:
            raise ValueError(f"unknown config '{c}', choose from {list(configs_all)}")

    images_per_config: Dict[str, torch.Tensor] = {}
    codes_per_config:  Dict[str, torch.Tensor] = {}

    for name in requested:
        c = configs_all[name]
        print(f"\n[run] config={name}  apply_k=set  apply_v=set")
        images, codes = run_one_config(
            trainer, sampler, cfg, name,
            apply_k=c["apply_k"], apply_v=c["apply_v"],
            labels=labels, seed=args.seed,
        )
        images_per_config[name] = images
        codes_per_config[name]  = codes
        # Per-config grid
        save_grid(images, os.path.join(args.out_dir, f"grid_{name}.png"),
                  ncol=min(args.nb_sample, 4))
        torch.save(images, os.path.join(args.out_dir, f"images_{name}.pt"))
        torch.save(codes,  os.path.join(args.out_dir, f"codes_{name}.pt"))
        print(f"[run] {name} done. saved images & codes.")

    # Side-by-side comparison grid (each row = one config).
    save_comparison_grid(images_per_config,
                         os.path.join(args.out_dir, "comparison_grid.png"),
                         ncol_per_row=args.nb_sample)
    print(f"[plot] saved {os.path.join(args.out_dir, 'comparison_grid.png')}")

    # ── Summary table ──
    if "none" not in requested:
        print("[warn] no vanilla baseline ('none') in --configs, skipping comparison")
        return
    images_v = images_per_config["none"]
    codes_v  = codes_per_config ["none"]
    summary_lines = [
        f"# KV substitution ablation summary",
        f"# vit={args.vit_size} img={args.img_size} steps={args.steps} "
        f"cfg-w={args.cfg_w} dtype={args.dtype} nb={args.nb_sample} seed={args.seed}",
        f"# gate: layer={args.gate_layer_min}..{args.gate_layer_max} "
        f"step={args.gate_step_min}..{args.gate_step_max}",
        "",
        f"{'config':<8} {'pixel_mse':>10} {'pixel_l1':>10} {'code_disagree':>14} "
        f"{'mse_min':>10} {'mse_med':>10} {'mse_max':>10}",
    ]
    print("\n" + summary_lines[-1])
    for name in requested:
        if name == "none":
            continue
        m = compute_metrics(images_v, codes_v,
                            images_per_config[name], codes_per_config[name])
        line = (f"{name:<8} {m['pixel_mse']:>10.5f} {m['pixel_l1']:>10.5f} "
                f"{m['code_disagree']:>14.4f} "
                f"{m['per_img_mse_min']:>10.5f} {m['per_img_mse_median']:>10.5f} "
                f"{m['per_img_mse_max']:>10.5f}")
        summary_lines.append(line)
        print(line)
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"\n[save] summary -> {os.path.join(args.out_dir, 'summary.txt')}")


if __name__ == "__main__":
    main()
