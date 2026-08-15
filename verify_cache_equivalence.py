"""等价性回归测试: 保存所有 cache 路径的生成结果 (图像 + 每 step 的 code)。

用途: 任何声称"只改实现、不改数值"的改动 (例如把 bool-mask indexing +
int(mask.sum().item()) 换成 gather/scatter 的 index 版本), 都应在改动前后各跑
一次本脚本, 再用 compare_cache_equivalence.py 比较, 确认逐位一致。

采样管线本身是确定的 (固定 seed 下同代码两次运行逐位一致), 所以位级比较是有效
判据 —— 任何非零差异都说明数值行为被改变了。

用法:
    python verify_cache_equivalence.py <out.pt> [dtype] [only]
      dtype : float32 (默认) | bfloat16
      only  : 逗号分隔的构型名子集 (缺省跑全部)
              baseline,ffn_N2,ffn_norefresh,attn_N2,attn_norefresh,
              layer_N2,layer_N4,layer_norefresh

    python verify_cache_equivalence.py ref_before.pt float32      # 改动前
    python verify_cache_equivalence.py ref_after.pt  float32      # 改动后
    python compare_cache_equivalence.py ref_before.pt ref_after.pt
"""
import os
import sys

# 走る前に全ての HALTON_* を消しておく (外部 export の混入防止)
for _k in ("HALTON_PARTIAL_UPDATE", "HALTON_ATTN_CACHE", "HALTON_LAYER_CACHE",
           "HALTON_CACHE_REFRESH_N", "HALTON_PARTIAL_START_LAYER",
           "HALTON_PARTIAL_END_LAYER"):
    os.environ.pop(_k, None)

import torch
from Utils.utils import load_args_from_file
from Trainer.cls_trainer import MaskGIT
from Sampler.halton_sampler import HaltonSampler

OUT = sys.argv[1]
DTYPE = sys.argv[2] if len(sys.argv) > 2 else "float32"
SEED = 1234
BATCH = 8

args = load_args_from_file("Config/base_cls2img.yaml")
args.device = torch.device("cuda")
args.vit_size = "large"
args.img_size = 384
args.compile = False
args.dtype = DTYPE
args.resume = True
args.vit_folder = f"./saved_networks/ImageNet_{args.img_size}_{args.vit_size}.pth"
model = MaskGIT(args)

labels = torch.LongTensor([1, 7, 282, 604, 724, 179, 681, 850]).to(args.device)


def set_env(partial, attn, layer, refresh_n, start_layer=None, end_layer=None):
    """全ての HALTON_* を毎回明示的に設定する (残留による汚染を防ぐ)。"""
    os.environ["HALTON_PARTIAL_UPDATE"] = "1" if partial else "0"
    os.environ["HALTON_ATTN_CACHE"] = "1" if attn else "0"
    os.environ["HALTON_LAYER_CACHE"] = "1" if layer else "0"
    os.environ["HALTON_CACHE_REFRESH_N"] = str(refresh_n)
    for key, val in (("HALTON_PARTIAL_START_LAYER", start_layer),
                     ("HALTON_PARTIAL_END_LAYER", end_layer)):
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(val)


# 生成側の分岐を全部踏むための構成一覧
CONFIGS = [
    # name,             partial, attn,  layer, N
    ("baseline",        False,   False, False, 0),
    ("ffn_N2",          True,    False, False, 2),
    ("ffn_norefresh",   True,    False, False, 0),
    ("attn_N2",         True,    True,  False, 2),
    ("attn_norefresh",  True,    True,  False, 0),
    ("layer_N2",        True,    False, True,  2),
    ("layer_N4",        True,    False, True,  4),
    ("layer_norefresh", True,    False, True,  0),
]

ONLY = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else None

out = {}
for name, partial, attn, layer, n in CONFIGS:
    if ONLY is not None and name not in ONLY:
        continue
    set_env(partial, attn, layer, n)
    # 各構成で RNG を同一状態から開始 -> 構成間の差は cache 機構のみ
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    sampler = HaltonSampler(sm_temp_min=1, sm_temp_max=1.2, temp_pow=1, temp_warmup=0,
                            w=2, sched_pow=2, step=32, randomize=True, top_k=-1)
    img, l_codes, _, _ = sampler(trainer=model, nb_sample=BATCH, labels=labels,
                                 verbose=False, partial_update=None)
    out[name] = {
        "img": img.detach().float().cpu(),
        "codes": torch.stack(l_codes).detach().cpu(),   # (step, b, h, w) int64
    }
    print(f"{name:16s} img{tuple(img.shape)} mean={img.mean().item():+.6f} "
          f"std={img.std().item():.6f} code_sum={out[name]['codes'].sum().item()}")

torch.save({"dtype": DTYPE, "seed": SEED, "batch": BATCH, "data": out}, OUT)
print(f"\nsaved -> {OUT}")
