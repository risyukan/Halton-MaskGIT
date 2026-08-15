"""生成示例图: 启用 layer-output cache (不改动模型架构), cache-refresh N=2。

与 generate_ffncache_N2.py / generate_attncache_N2.py 的区别: 这里走的是
layer-output cache 方案 —— 每层缓存整层输出, partial 步只重算 active token
(Q 只取 active, K/V 取全部, inactive 的 K/V 来自缓存的上一步本层输入), FFN 也只
算 active; inactive 位置直接沿用缓存的整层输出, 最后把缓存里 active 位置更新。
ffn cache 与 attention cache 在此方案下都关闭 (代码保留, 只是不走)。

对应 env:
    HALTON_PARTIAL_UPDATE=1    # 采样器把 active_mask(U_t) 传给 transformer
    HALTON_LAYER_CACHE=1       # 走 layer-output cache 分支 (关掉 ffn/attn cache)
    HALTON_CACHE_REFRESH_N=2   # 每 2 个 gated step 做一次 full 步刷新缓存
    (不设置 HALTON_ATTN_CACHE -> attention cache 关闭)
"""
import os

# 必须在 import 网络/采样器之前设好 (forward 里按环境变量分支)。
os.environ["HALTON_PARTIAL_UPDATE"] = "1"
os.environ["HALTON_LAYER_CACHE"] = "1"
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

# partial_update=True => 启用 active_mask, 走 layer-output cache 路径。
gen_images = sampler(trainer=model, nb_sample=8, labels=labels,
                     verbose=True, partial_update=True)[0]

out = "gen_layercache_N2.png"
grid = make_grid((gen_images.clamp(-1, 1) + 1) / 2.0, nrow=4, padding=2)
save_image(grid, out)
print(f"saved -> {out}")
