"""生成示例图: 仅启用 FFN cache (不改动模型架构), cache-refresh N=2。

与 generate_attncache_N2.py 的区别: 只走 FFN cache, 关闭 attention cache
(attention 保持 full update, baseline 不变), 用于论文里提案手法的例图。

对应 env:
    HALTON_PARTIAL_UPDATE=1    # active-only FFN + cached_ffn_delta
    HALTON_CACHE_REFRESH_N=2   # 每 2 个 gated step 刷新一次缓存
    (不设置 HALTON_ATTN_CACHE -> attention 仍是全 token, 只有 FFN 用 cache)
"""
import os

# 必须在 import 网络/采样器之前设好 (forward 里按环境变量分支)。
os.environ["HALTON_PARTIAL_UPDATE"] = "1"
os.environ["HALTON_CACHE_REFRESH_N"] = "2"
os.environ.pop("HALTON_ATTN_CACHE", None)   # 显式关闭 attention cache

import torch
from torchvision.utils import make_grid, save_image
from Utils.utils import load_args_from_file
from Trainer.cls_trainer import MaskGIT
from Sampler.halton_sampler import HaltonSampler

config_path = "Config/base_cls2img.yaml"
args = load_args_from_file(config_path)

args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.vit_size = "large"          # large-384 是最好(也最慢)的网络
args.img_size = 384
args.compile = False
args.dtype = "float32"
args.resume = True
args.vit_folder = f"./saved_networks/ImageNet_{args.img_size}_{args.vit_size}.pth"

# 本地已存在权重, 不再走 hf_hub_download。
model = MaskGIT(args)

sampler = HaltonSampler(sm_temp_min=1, sm_temp_max=1.2, temp_pow=1, temp_warmup=0, w=2,
                        sched_pow=2, step=32, randomize=True, top_k=-1)

# [goldfish, chicken, tiger cat, hourglass, ship, dog, race car, airliner]
labels = torch.LongTensor([1, 7, 282, 604, 724, 179, 681, 850]).to(args.device)

# partial_update=True => 启用 active_mask, 只有 FFN 走 active-only + cached-delta 路径。
gen_images = sampler(trainer=model, nb_sample=8, labels=labels,
                     verbose=True, partial_update=True)[0]

out = "gen_ffncache_N2.png"
grid = make_grid((gen_images.clamp(-1, 1) + 1) / 2.0, nrow=4, padding=2)
save_image(grid, out)
print(f"saved -> {out}")
