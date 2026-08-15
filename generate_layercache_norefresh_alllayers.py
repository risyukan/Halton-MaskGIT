"""生成示例图: layer-output cache, layer 全开 (0..23), 完全不刷新 (N=0)。

与 generate_layercache_N4_alllayers.py 的区别:
  - HALTON_CACHE_REFRESH_N=0  (纯部分更新, 0/26 refresh, 缓存从不重写)
对应 FID 评测: launch/eval_fid_large384_layercache_alllayers_norefresh.sh
  (FID 10.773 / IS 153.9, 理论 FLOPs 加速 2.86x —— 极限加速/质量下限展示用)

对应 env:
    HALTON_PARTIAL_UPDATE=1    # 采样器把 active_mask(U_t) 传给 transformer
    HALTON_LAYER_CACHE=1       # 走 layer-output cache 分支 (关掉 ffn/attn cache)
    HALTON_CACHE_REFRESH_N=0   # 0 => 不做周期性刷新
    HALTON_PARTIAL_START_LAYER=0 / HALTON_PARTIAL_END_LAYER=23
    (不设置 HALTON_ATTN_CACHE -> attention cache 关闭)
"""
import os

# 必须在 import 网络/采样器之前设好 (forward 里按环境变量分支)。
os.environ["HALTON_PARTIAL_UPDATE"] = "1"
os.environ["HALTON_LAYER_CACHE"] = "1"
os.environ["HALTON_CACHE_REFRESH_N"] = "0"
os.environ["HALTON_PARTIAL_START_LAYER"] = "0"
os.environ["HALTON_PARTIAL_END_LAYER"] = "23"
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

out = "gen_layercache_norefresh_alllayers.png"
grid = make_grid((gen_images.clamp(-1, 1) + 1) / 2.0, nrow=4, padding=2)
save_image(grid, out)
print(f"saved -> {out}")
