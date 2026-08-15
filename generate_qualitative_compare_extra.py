"""定性比較 (追加 4 クラス) 用の画像を生成。

generate_qualitative_compare.py と同一のパイプライン・sampler 設定で、
別の 4 クラスを生成する。cache 機構だけを変えた 3 設定を同一 seed / label で:
    baseline : partial_update=False                (全 FFN, cache なし)
    norefresh: partial_update=True,  refresh_n=0    (純部分更新, cache 更新なし)
    n4       : partial_update=True,  refresh_n=4    (N=4 でリフレッシュ)

出力は statics/qual_compare_extra/images.pt (組版は compose 側)。
"""
import os

os.environ.pop("HALTON_ATTN_CACHE", None)
os.environ.pop("HALTON_PARTIAL_END_LAYER", None)

import numpy as np
import torch
from torchvision.utils import save_image
from Utils.utils import load_args_from_file
from Trainer.cls_trainer import MaskGIT
from Sampler.halton_sampler import HaltonSampler

OUT_DIR = "statics/qual_compare_extra"
os.makedirs(OUT_DIR, exist_ok=True)

# 追加 4 クラス (テクスチャ/ディテールが豊富で cache 退化が見えやすいもの)
LABELS = [130, 207, 963, 973]
LABEL_NAMES = ["flamingo", "golden retriever", "pizza", "coral reef"]

SEEDS = [1000, 1001, 1002, 1003, 1004, 1005]

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
        imgs = (gen.clamp(-1, 1) + 1) / 2.0
        store[seed][name] = imgs.cpu()

        for i, cls in enumerate(LABEL_NAMES):
            save_image(imgs[i], os.path.join(OUT_DIR, f"seed{seed}_{name}_{cls.replace(' ','')}.png"))
        print(f"[seed {seed}] {name} (refresh_n={refresh_n}) done.")

torch.save({"store": store, "labels": LABELS, "label_names": LABEL_NAMES,
            "seeds": SEEDS, "configs": [c[0] for c in CONFIGS]},
           os.path.join(OUT_DIR, "images.pt"))
print("\nsaved tensors ->", os.path.join(OUT_DIR, "images.pt"))
