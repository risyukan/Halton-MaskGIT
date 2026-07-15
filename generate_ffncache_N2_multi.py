"""多轮生成: 仅 FFN cache (不改动模型架构), cache-refresh N=2。

跑 N_ROUNDS 轮, 每轮固定不同 seed, 各存一张 8-图 grid, 便于挑最好看的一轮。
"""
import os

os.environ["HALTON_PARTIAL_UPDATE"] = "1"
os.environ["HALTON_CACHE_REFRESH_N"] = "2"
os.environ.pop("HALTON_ATTN_CACHE", None)   # 关闭 attention cache

import numpy as np
import torch
from torchvision.utils import make_grid, save_image
from Utils.utils import load_args_from_file
from Trainer.cls_trainer import MaskGIT
from Sampler.halton_sampler import HaltonSampler

N_ROUNDS = 8
OUT_DIR = "statics/ffncache_N2_rounds"
os.makedirs(OUT_DIR, exist_ok=True)

config_path = "Config/base_cls2img.yaml"
args = load_args_from_file(config_path)
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.vit_size = "large"
args.img_size = 384
args.compile = False
args.dtype = "float32"
args.resume = True
args.vit_folder = f"./saved_networks/ImageNet_{args.img_size}_{args.vit_size}.pth"

model = MaskGIT(args)

sampler = HaltonSampler(sm_temp_min=1, sm_temp_max=1.2, temp_pow=1, temp_warmup=0, w=2,
                        sched_pow=2, step=32, randomize=True, top_k=-1)

# [goldfish, chicken, tiger cat, hourglass, ship, dog, race car, airliner]
labels = torch.LongTensor([1, 7, 282, 604, 724, 179, 681, 850]).to(args.device)

for r in range(N_ROUNDS):
    seed = 1000 + r
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    gen_images = sampler(trainer=model, nb_sample=8, labels=labels,
                         verbose=True, partial_update=True)[0]
    grid = make_grid((gen_images.clamp(-1, 1) + 1) / 2.0, nrow=4, padding=2)
    out = os.path.join(OUT_DIR, f"round_{r:02d}_seed{seed}.png")
    save_image(grid, out)
    print(f"[round {r}] seed={seed} saved -> {out}")

print("done.")
