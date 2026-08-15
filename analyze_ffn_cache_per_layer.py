"""
Per-layer causal probe: how much does *actually caching* the previous-step FFN
delta on inactive tokens hurt the final output — layer by layer?

Unlike analyze_ffn_delta_stability.py (which only measures cos/drift *similarity*
of the delta), this script measures the CAUSAL effect: at every decoding step we
run counterfactual forwards where, for a chosen set of layers, inactive tokens
reuse the previous step's (gated) FFN delta instead of recomputing it, and we
compare the resulting CFG-combined logits against the true (all-fresh) logits.

Per step t (>=1), sharing the same input `code`:
  1. TRUE pass  (all layers fresh)  -> reference logits, and record each layer's
     true delta as the cache for the NEXT step.
  2. For each probe config C (a set of layers): one forward where layers in C
     reuse prev-step delta on inactive tokens -> logits_C, compared to reference.

Metrics on inactive tokens (mean over steps/samples/tokens):
  flip = P(argmax(logits_C) != argmax(logits_true))     # token-level damage
  KL   = KL( softmax(logits_true) || softmax(logits_C) )
  relL2= || logits_C - logits_true ||_2 / || logits_true ||_2

Probe configs: every single layer {l}, plus ALL / no_first / no_last / no_ends /
mid_only, so we can read directly whether first/last layers must be excluded.

Does NOT modify any model source — Block.forward is monkey-patched at runtime.

Output:
  <out-dir>/ffn_cache_per_layer.png   per-layer flip/KL/relL2 + config bars
  <out-dir>/ffn_cache_per_layer.npz   raw aggregated arrays
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


# ---- runtime state shared with the patched forward -------------------------
_S = {
    "mode": "true",          # "true" | "probe"
    "cache_layers": set(),   # layers that reuse prev delta (probe mode only)
    "inact": None,           # (B_all, seq) bool — inactive token positions to cache
}


def _patched_forward(self, x, cond, mask=None, active_idx=None):
    """Mirror of Block.forward (attention always fresh; only FFN is probed)."""
    l = self._probe_idx
    gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.mlp(cond).chunk(6, dim=1)
    x = x + alpha1.unsqueeze(1) * self.attn(
        modulate(self.ln1(x), gamma1, beta1), mask=mask, active_idx=None,
    )
    true_delta = alpha2.unsqueeze(1) * self.ff(modulate(self.ln2(x), gamma2, beta2))

    if (_S["mode"] == "probe" and l in _S["cache_layers"]
            and self._prev_delta is not None
            and self._prev_delta.shape == true_delta.shape):
        delta = true_delta.clone()
        m = _S["inact"]
        delta[m] = self._prev_delta[m]
    else:
        delta = true_delta

    if _S["mode"] == "true":
        # stash this step's TRUE delta; committed to _prev_delta after the probes
        self._curr_delta = true_delta.detach()
    return x + delta


def patch_blocks(transformer):
    for i, blk in enumerate(transformer.layers):
        blk._probe_idx = i
        blk._prev_delta = None
        blk._curr_delta = None
        blk.forward = _patched_forward.__get__(blk, Block)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",    type=str, default="Config/base_cls2img.yaml")
    p.add_argument("--vit-size",  type=str, default="small")
    p.add_argument("--img-size",  type=int, default=384)
    p.add_argument("--steps",     type=int, default=32)
    p.add_argument("--nb-sample", type=int, default=4)
    p.add_argument("--cfg-w",     type=float, default=2.0)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--dtype",     type=str, default="bfloat16")
    p.add_argument("--out-dir",   type=str, default="statics/ffn_cache_probe")
    return p.parse_args()


def build_configs(L):
    """Return ordered list of (name, frozenset(layers))."""
    cfgs = [(f"L{l}", frozenset({l})) for l in range(L)]
    cfgs += [
        ("ALL",      frozenset(range(L))),
        ("no_first", frozenset(range(1, L))),
        ("no_last",  frozenset(range(0, L - 1))),
        ("no_ends",  frozenset(range(1, L - 1))),
        ("mid_only", frozenset(range(3, L - 2))),   # ~ current default gate
    ]
    return cfgs


@torch.no_grad()
def run(args):
    cfg = load_args_from_file(args.config)
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.vit_size, cfg.img_size = args.vit_size, args.img_size
    cfg.compile, cfg.dtype, cfg.resume = False, args.dtype, True
    cfg.vit_folder = f"./saved_networks/ImageNet_{cfg.img_size}_{cfg.vit_size}.pth"
    cfg.data_folder = cfg.eval_folder = cfg.writer_log = ""
    cfg.debug = True

    if args.seed >= 0:
        torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)

    print(f"[init] MaskGIT vit={cfg.vit_size} img={cfg.img_size} dtype={cfg.dtype}")
    trainer = MaskGIT(cfg)
    transformer = trainer.vit.module if hasattr(trainer.vit, "module") else trainer.vit
    patch_blocks(transformer.transformer)
    L = len(transformer.transformer.layers)
    register = transformer.register
    print(f"[init] patched {L} blocks, register={register}")

    sampler = HaltonSampler(sm_temp_min=1, sm_temp_max=1.2, temp_pow=1, temp_warmup=0,
                            w=args.cfg_w, sched_pow=2, step=args.steps,
                            randomize=False, top_k=-1)

    input_size = trainer.input_size
    nb = args.nb_sample
    N = input_size * input_size
    T = args.steps

    demo_labels = [1, 7, 282, 604, 724, 179, 681, 850, 850]
    labels = torch.LongTensor(demo_labels[:nb]).to(cfg.device)
    trainer.vit.eval()
    drop = torch.ones(nb, dtype=torch.bool, device=cfg.device)
    code = torch.full((nb, input_size, input_size), cfg.mask_value,
                      dtype=torch.long, device=cfg.device)
    sampler.compute_schedule(input_size, nb_sample=nb)  # populates basic_halton_mask
    halton_mask = sampler.basic_halton_mask.clone().unsqueeze(0).expand(nb, N, 2)

    configs = build_configs(L)
    # accumulators per config: flip / KL / relL2 summed over inactive tokens
    acc = {name: dict(flip=0.0, kl=0.0, rl2=0.0, n=0.0) for name, _ in configs}

    def combined_logits():
        logit = trainer.vit(
            torch.cat([code.clone(), code.clone()], dim=0),
            torch.cat([labels, labels], dim=0),
            torch.cat([~drop, drop], dim=0),
            active_mask=None,
        )
        lc, lu = torch.chunk(logit, 2, dim=0)
        return ((1 + sampler.w) * lc - sampler.w * lu).float()  # (nb, N, V)

    prev_r = 0
    for t in range(T):
        ratio = (t + 1) / T
        r = 1 - (torch.arccos(torch.tensor(ratio)) / (math.pi * 0.5))
        r = max(t + 1, int(r * (input_size ** 2)))
        _u = halton_mask[:, prev_r:r]
        U_t = torch.zeros(nb, input_size, input_size, dtype=torch.bool)
        for i in range(nb):
            U_t[i, _u[i, :, 0], _u[i, :, 1]] = True
        U_flat = U_t.view(nb, N)                       # newly released this step
        inact_real = ~U_flat                           # (nb, N) inactive tokens

        # full inactive mask over the 2*nb batch incl. register cols (register=fresh)
        seq = N + register
        inact_full = torch.zeros(2 * nb, seq, dtype=torch.bool, device=cfg.device)
        ir = inact_real.to(cfg.device)
        inact_full[:nb, :N] = ir
        inact_full[nb:, :N] = ir
        _S["inact"] = inact_full

        # ---- TRUE pass: reference logits + stash each layer's true delta -----
        _S["mode"] = "true"
        with trainer.autocast:
            logit_true = combined_logits()

        # probes only make sense once a previous-step delta exists (t >= 1)
        if t >= 1:
            _S["mode"] = "probe"
            pt = torch.softmax(logit_true, dim=-1)                    # (nb, N, V)
            argmax_true = logit_true.argmax(-1)                       # (nb, N)
            true_norm = logit_true.norm(dim=-1)                       # (nb, N)
            im = inact_real.to(cfg.device)                           # (nb, N) bool
            n_inact = int(im.sum().item())
            for name, layers in configs:
                _S["cache_layers"] = layers
                with trainer.autocast:
                    logit_c = combined_logits()
                flip = (logit_c.argmax(-1) != argmax_true) & im
                kl = (pt * (torch.log(pt + 1e-12)
                            - torch.log_softmax(logit_c, dim=-1))).sum(-1)   # (nb,N)
                rl2 = (logit_c - logit_true).norm(dim=-1) / (true_norm + 1e-12)
                a = acc[name]
                a["flip"] += float(flip.sum().item())
                a["kl"]   += float(kl[im].sum().item())
                a["rl2"]  += float(rl2[im].sum().item())
                a["n"]    += n_inact

        # ---- commit true deltas as next step's cache; advance trajectory -----
        for blk in transformer.transformer.layers:
            blk._prev_delta = blk._curr_delta
        _temp = sampler.temperature[t] ** 1
        prob = torch.softmax(logit_true * _temp, dim=-1)
        pred = torch.distributions.Categorical(probs=prob).sample()
        code[U_t.to(cfg.device)] = pred.view(nb, input_size, input_size)[U_t.to(cfg.device)]
        print(f"[step {t:02d}/{T}] released {U_flat.sum().item()//nb}/sample  r={r}")
        prev_r = r

    return configs, acc, L


def plot(configs, acc, L, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def get(name, key):
        a = acc[name]
        return a[key] / a["n"] if a["n"] > 0 else np.nan

    layer_flip = np.array([get(f"L{l}", "flip") for l in range(L)])
    layer_kl   = np.array([get(f"L{l}", "kl")   for l in range(L)])
    layer_rl2  = np.array([get(f"L{l}", "rl2")  for l in range(L)])

    combo_names = ["ALL", "no_first", "no_last", "no_ends", "mid_only"]
    combo_flip = np.array([get(n, "flip") for n in combo_names])
    combo_kl   = np.array([get(n, "kl")   for n in combo_names])

    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))

    a = ax[0, 0]
    a.plot(layer_flip * 100, marker="o", color="C3")
    a.axvspan(-0.5, 2.5, color="gray", alpha=0.15)
    a.axvspan(L - 2.5, L - 0.5, color="gray", alpha=0.15)
    a.set_xlabel("layer index"); a.set_ylabel("argmax flip rate (%)")
    a.set_title("per-layer cache damage — token flip rate"); a.grid(True, alpha=0.3)

    a = ax[0, 1]
    a.plot(layer_kl, marker="o", color="C0")
    a.axvspan(-0.5, 2.5, color="gray", alpha=0.15)
    a.axvspan(L - 2.5, L - 0.5, color="gray", alpha=0.15)
    a.set_xlabel("layer index"); a.set_ylabel("KL(true || cached)")
    a.set_title("per-layer cache damage — output KL"); a.grid(True, alpha=0.3)

    a = ax[1, 0]
    a.plot(layer_rl2, marker="o", color="C2")
    a.set_xlabel("layer index"); a.set_ylabel("rel L2 of logits")
    a.set_title("per-layer cache damage — logit rel-L2"); a.grid(True, alpha=0.3)

    a = ax[1, 1]
    x = np.arange(len(combo_names))
    a.bar(x - 0.2, combo_flip * 100, width=0.4, label="flip rate (%)", color="C3")
    a.bar(x + 0.2, combo_kl, width=0.4, label="KL", color="C0")
    a.set_xticks(x); a.set_xticklabels(combo_names, rotation=20)
    a.set_title("cache configs — is separating first/last worth it?")
    a.legend(); a.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    p = os.path.join(out_dir, "ffn_cache_per_layer.png")
    plt.savefig(p, dpi=120); plt.close(fig)
    print("[plot] saved", p)

    print("\n=== per-layer cache damage (inactive tokens) ===")
    print("layer  flip%    KL       relL2")
    for l in range(L):
        print(f"{l:5d}  {layer_flip[l]*100:6.2f}  {layer_kl[l]:7.4f}  {layer_rl2[l]:7.4f}")
    print("\n=== config comparison ===")
    print("config      flip%    KL")
    for n in combo_names:
        print(f"{n:10s}  {get(n,'flip')*100:6.2f}  {get(n,'kl'):7.4f}")

    np.savez_compressed(
        os.path.join(out_dir, "ffn_cache_per_layer.npz"),
        layer_flip=layer_flip, layer_kl=layer_kl, layer_rl2=layer_rl2,
        combo_names=np.array(combo_names),
        combo_flip=combo_flip, combo_kl=combo_kl,
    )


def main():
    args = parse_args()
    configs, acc, L = run(args)
    plot(configs, acc, L, args.out_dir)


if __name__ == "__main__":
    main()
