"""定性比較 (ImageNet 384x384, large 模型) 用の画像を生成。

3 つの設定を同一 seed / 同一 label で生成し、cache 機構だけを変える:
    baseline : partial_update=False                (全 FFN, cache なし)
    norefresh: partial_update=True,  refresh_n=0    (純部分更新, cache 更新なし)
    n4       : partial_update=True,  refresh_n=4    (N=4 でリフレッシュ)

sampler ハイパラは 3 設定で完全に一致させ、cache 経路だけを差にする
(poster_strips の large 設定に合わせる: w=0.5, sm_temp_max=1.2, temp_warmup=0)。

生成した個々の画像 tensor を .pt に保存し、組版は compose_qualitative_compare.py
側で行う (図の調整で毎回再生成しないため)。
"""
import os

# attention cache は使わない (FFN cache のみ)。
os.environ.pop("HALTON_ATTN_CACHE", None)
os.environ.pop("HALTON_PARTIAL_END_LAYER", None)  # large は default gate (3..21)

import numpy as np
import torch
from torchvision.utils import save_image
from Utils.utils import load_args_from_file
from Trainer.cls_trainer import MaskGIT
from Sampler.halton_sampler import HaltonSampler

OUT_DIR = "statics/qual_compare"
os.makedirs(OUT_DIR, exist_ok=True)

# 4 クラス (行), 見分けやすいもの。goldfish(1), tiger cat(282), ship(724), race car(681)
LABELS = [1, 282, 724, 681]
LABEL_NAMES = ["goldfish", "tiger cat", "ship", "race car"]

# 複数 seed 生成して後で一番きれいな seed を選べるように。
SEEDS = [1000, 1001, 1002, 1003, 1004, 1005]

# 3 設定: (name, partial_update, refresh_n)
CONFIGS = [
    ("baseline",  False, 0),
    ("norefresh", True,  0),
    ("n4",        True,  4),
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = load_args_from_file("Config/base_cls2img.yaml")
args.device = device
args.vit_size = "large"
args.img_size = 384
args.compile = False
args.dtype = "float32"
args.resume = True
args.vit_folder = f"./saved_networks/ImageNet_{args.img_size}_{args.vit_size}.pth"

model = MaskGIT(args)
model.vit.eval()

sampler = HaltonSampler(sm_temp_min=1, sm_temp_max=1.2, temp_pow=1, temp_warmup=0,
                        w=0.5, sched_pow=2, step=32, randomize=True, top_k=-1)

labels = torch.LongTensor(LABELS).to(device)

# store[seed][config] = tensor (N,3,H,W) in [0,1]
store = {}
for seed in SEEDS:
    store[seed] = {}
    for name, partial, refresh_n in CONFIGS:
        os.environ["HALTON_CACHE_REFRESH_N"] = str(refresh_n)

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        gen = sampler(trainer=model, nb_sample=len(LABELS), labels=labels,
                      verbose=True, partial_update=partial)[0]
        imgs = (gen.clamp(-1, 1) + 1) / 2.0  # -> [0,1]
        store[seed][name] = imgs.cpu()

        # 個別 png も保存 (目視確認用)
        for i, cls in enumerate(LABEL_NAMES):
            save_image(imgs[i], os.path.join(OUT_DIR, f"seed{seed}_{name}_{cls.replace(' ','')}.png"))
        print(f"[seed {seed}] {name} (refresh_n={refresh_n}) done.")

torch.save({"store": store, "labels": LABELS, "label_names": LABEL_NAMES,
            "seeds": SEEDS, "configs": [c[0] for c in CONFIGS]},
           os.path.join(OUT_DIR, "images.pt"))
print("\nsaved tensors ->", os.path.join(OUT_DIR, "images.pt"))
