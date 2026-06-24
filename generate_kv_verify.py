"""Visually verify the FFN_N2 + KV_N2 cache config (the FID 2.6638 run).

Generates the SAME 8 class labels with the SAME seed under two configs and
saves them side by side so the cache approximation can be eye-balled:

  (A) baseline      : partial_update=False, no FFN/KV cache  (== FID 2.541 run)
  (B) ffn2_kv2      : partial_update=True, FFN refresh N=2 + KV refresh N=2
                      (== results/halton_large384_kvcache_sweep.txt, FID 2.6638)

All sampler hyper-params match the FID eval pipeline (cls_trainer builds
self.sampler from args): halton / step32 / cfg_w0.5 / sm_temp1.0 /
temp_warmup1 / sched_pow2 / top_k-1 / randomize False / fp32.

Run on a single free GPU, e.g.:
  CUDA_VISIBLE_DEVICES=1 python generate_kv_verify.py
"""
import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from Utils.utils import load_args_from_file
from Trainer.cls_trainer import MaskGIT

SEED = 42
LABELS = [1, 7, 282, 604, 724, 179, 681, 850]  # goldfish, chicken, tiger cat,
#                                                hourglass, ship, dog, race car, airliner
OUT_DIR = "results/kv_verify"
os.makedirs(OUT_DIR, exist_ok=True)

# ---- args: match the FID eval pipeline exactly --------------------------
args = load_args_from_file("Config/base_cls2img.yaml")
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.vit_size = "large"
args.img_size = 384
args.compile = False
args.dtype = "float32"
args.resume = True
args.vit_folder = f"./saved_networks/ImageNet_{args.img_size}_{args.vit_size}.pth"
args.vqgan_folder = "./saved_networks/vq_ds16_c2i.pt"
# sampler hyper-params = those used by the FID run (eval_fid_large384_ffn2_kv2.sh)
args.sampler = "halton"
args.step = 32
args.cfg_w = 0.5
args.sm_temp = 1.0
args.sm_temp_min = 1.0
args.temp_warmup = 1
args.sched_pow = 2.0
args.top_k = -1
args.randomize = False

model = MaskGIT(args)                 # builds model.sampler from args
sampler = model.sampler
labels = torch.LongTensor(LABELS).to(args.device)


def seeded():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def save_grid(batch, path, title):
    batch = ((batch + 1) / 2.0).clamp(0, 1)
    grid = make_grid(batch, nrow=4, padding=2).permute(1, 2, 0).cpu().numpy()
    plt.figure(figsize=(8, 4.5))
    plt.imshow(grid)
    plt.axis("off")
    plt.title(title, fontsize=11)
    plt.savefig(path, bbox_inches="tight", pad_inches=0.05, dpi=150)
    plt.close()
    print(f"saved -> {path}")


# ---- (A) baseline: no cache --------------------------------------------
os.environ["HALTON_PARTIAL_UPDATE"] = "0"
os.environ["HALTON_CACHE_REFRESH_N"] = "0"
os.environ["HALTON_KV_REFRESH_N"] = "0"
seeded()
img_base = sampler(trainer=model, nb_sample=len(LABELS), labels=labels,
                   verbose=True, partial_update=False)[0]
save_grid(img_base, f"{OUT_DIR}/A_baseline.png",
          "(A) baseline (full FFN+KV)  —  FID 2.541")

# ---- (B) ffn2_kv2: FFN refresh N=2 + KV refresh N=2 ---------------------
os.environ["HALTON_PARTIAL_UPDATE"] = "1"
os.environ["HALTON_CACHE_REFRESH_N"] = "2"
os.environ["HALTON_KV_REFRESH_N"] = "2"
seeded()
img_kv = sampler(trainer=model, nb_sample=len(LABELS), labels=labels,
                 verbose=True, partial_update=True)[0]
save_grid(img_kv, f"{OUT_DIR}/B_ffn2_kv2.png",
          "(B) FFN N=2 + KV N=2 cache  —  FID 2.6638")

# ---- side-by-side + per-image L1 diff -----------------------------------
diff = (img_kv - img_base).abs()
print(f"\nper-image mean|Δ| (pixel space, [-1,1]):")
for lab, d in zip(LABELS, diff.flatten(1).mean(1).tolist()):
    print(f"  class {lab:4d} : {d:.4f}")
print(f"overall mean|Δ| = {diff.mean().item():.4f}")

both = torch.cat([img_base, img_kv], dim=0)
save_grid(both, f"{OUT_DIR}/AB_compare.png",
          "top: baseline (FID 2.541)   |   bottom: FFN2+KV2 (FID 2.6638)")
print("\nDONE. Compare results/kv_verify/AB_compare.png")
