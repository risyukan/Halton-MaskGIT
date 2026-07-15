"""海报用: 为 L/B/S 三个配置各生成 4 张连在一起的图像 (1x4 strip)。

不改动任何模型/采样器代码, 全部通过环境变量 + 采样器参数控制, 使每个配置
与其 FID sweep 严格对应:

    Large : N=4, cfg_w=0.5, layer gate 3..21 (默认), sm_temp_max=1.2, temp_warmup=0
    Base  : N=9, cfg_w=0.7, layer gate 3..9  (END_LAYER=9), sm_temp=1.0, temp_warmup=1
    Small : N=9, cfg_w=1.0, layer gate 3..9  (END_LAYER=9), sm_temp=1.0, temp_warmup=1

三个配置公用相同的 step=32 / sched_pow=2 / top_k=-1 / seed, 只走 FFN cache
(attention 保持全量, 不设 HALTON_ATTN_CACHE)。

每个配置跑 N_ROUNDS 个 seed, 各存一张 4-图 strip, 便于挑最好看的一轮贴海报。
"""
import os

# 只启用 FFN partial-update + cache (attention 全量)。partial_update 在采样时显式传 True,
# HALTON_CACHE_REFRESH_N / HALTON_PARTIAL_END_LAYER 在采样时读取, 因此每个配置循环内动态设置。
os.environ.pop("HALTON_ATTN_CACHE", None)

import numpy as np
import torch
from torchvision.utils import make_grid, save_image
from Utils.utils import load_args_from_file
from Trainer.cls_trainer import MaskGIT
from Sampler.halton_sampler import HaltonSampler

N_ROUNDS = 6
OUT_ROOT = "statics/poster_strips"
os.makedirs(OUT_ROOT, exist_ok=True)

# 4 个可辨识度高的 ImageNet 类, 三个模型共用, 便于海报排版一致。
# goldfish(1), tiger cat(282), ship(724), race car(681)
LABELS = [1, 282, 724, 681]

CONFIGS = [
    dict(name="large_N4_cfg05", vit_size="large", refresh_n=4, w=0.5,
         end_layer=None, sm_temp_max=1.2, temp_warmup=0),
    dict(name="base_N9_cfg07",  vit_size="base",  refresh_n=9, w=0.7,
         end_layer=9,   sm_temp_max=1.0, temp_warmup=1),
    dict(name="small_N9_cfg10", vit_size="small", refresh_n=9, w=1.0,
         end_layer=9,   sm_temp_max=1.0, temp_warmup=1),
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(vit_size):
    args = load_args_from_file("Config/base_cls2img.yaml")
    args.device = device
    args.vit_size = vit_size
    args.img_size = 384
    args.compile = False
    args.dtype = "float32"
    args.resume = True
    args.vit_folder = f"./saved_networks/ImageNet_{args.img_size}_{vit_size}.pth"
    m = MaskGIT(args)
    m.vit.eval()
    return m


for cfg in CONFIGS:
    print(f"\n=== {cfg['name']} (N={cfg['refresh_n']}, cfg_w={cfg['w']}) ===")
    # 每个配置的 cache-refresh 间隔 / 层 gate 通过 env 控制 (采样时读取)。
    os.environ["HALTON_CACHE_REFRESH_N"] = str(cfg["refresh_n"])
    if cfg["end_layer"] is None:
        os.environ.pop("HALTON_PARTIAL_END_LAYER", None)
    else:
        os.environ["HALTON_PARTIAL_END_LAYER"] = str(cfg["end_layer"])

    model = build_model(cfg["vit_size"])
    sampler = HaltonSampler(sm_temp_min=1, sm_temp_max=cfg["sm_temp_max"], temp_pow=1,
                            temp_warmup=cfg["temp_warmup"], w=cfg["w"],
                            sched_pow=2, step=32, randomize=True, top_k=-1)

    labels = torch.LongTensor(LABELS).to(device)
    out_dir = os.path.join(OUT_ROOT, cfg["name"])
    os.makedirs(out_dir, exist_ok=True)

    for r in range(N_ROUNDS):
        seed = 1000 + r
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        gen = sampler(trainer=model, nb_sample=len(LABELS), labels=labels,
                      verbose=True, partial_update=True)[0]
        grid = make_grid((gen.clamp(-1, 1) + 1) / 2.0, nrow=len(LABELS), padding=2)
        out = os.path.join(out_dir, f"round_{r:02d}_seed{seed}.png")
        save_image(grid, out)
        print(f"  [round {r}] seed={seed} -> {out}")

    del model
    torch.cuda.empty_cache()

print("\ndone. strips in", OUT_ROOT)
