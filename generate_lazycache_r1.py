"""定性对比图: baseline vs LazyMAR Token Cache (r=1, 全部 24 层)。

同一个 seed / 同一批类别各跑一次, 拼成上下两行, 用来肉眼看这套近似有没有把
图像质量打崩 —— latency / FLOPs 的数字见 bench_latency.py 与
flops_lazy_token_cache.py。

对应 env (lazy 档):
    HALTON_PARTIAL_UPDATE=1     # 采样器下发 active_mask = U_t ∪ U_{t-1}
    HALTON_LAZY_CACHE=1         # 走 LazyMAR Token Cache 路径
    HALTON_LAZY_CACHE_RATIO=1.0 # r=1 -> 不打分, active 就是强制集合
    (层区间不设 -> 默认 0..depth-1)
"""
import os
import torch
from torchvision.utils import make_grid, save_image

from Utils.utils import load_args_from_file
from Trainer.cls_trainer import MaskGIT
from Sampler.halton_sampler import HaltonSampler

args = load_args_from_file("Config/base_cls2img.yaml")
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.vit_size = "large"
args.img_size = 384
args.compile = False
args.dtype = "float32"
args.resume = True
args.vit_folder = f"./saved_networks/ImageNet_{args.img_size}_{args.vit_size}.pth"
model = MaskGIT(args)

labels = torch.LongTensor([1, 7, 282, 604, 724, 179, 681, 850]).to(args.device)


def gen(lazy):
    os.environ["HALTON_PARTIAL_UPDATE"] = "1" if lazy else "0"
    os.environ["HALTON_LAZY_CACHE"] = "1" if lazy else "0"
    os.environ["HALTON_LAZY_CACHE_RATIO"] = "1.0"
    os.environ["HALTON_ATTN_CACHE"] = "0"
    os.environ["HALTON_CACHE_REFRESH_N"] = "0"
    for k in ("HALTON_LAZY_START_LAYER", "HALTON_LAZY_END_LAYER",
              "HALTON_PARTIAL_START_LAYER", "HALTON_PARTIAL_END_LAYER"):
        os.environ.pop(k, None)
    sampler = HaltonSampler(sm_temp_min=1, sm_temp_max=1.2, temp_pow=1, temp_warmup=0,
                            w=2, sched_pow=2, step=32, randomize=True, top_k=-1)
    torch.manual_seed(1234)
    img = sampler(trainer=model, nb_sample=8, labels=labels,
                  verbose=True, partial_update=None)[0]
    if lazy:
        st = getattr(model.vit, "module", model.vit).get_lazy_stats()
        print(f"  lazy stats: partial 步 k/N={st['partial_active_ratio']:.4f} "
              f"({st['partial_calls']} 步), 全量步 {st['full_calls']}, "
              f"打分次数={st['scored_calls']}, kv_miss={st['kv_miss']}")
    return img


base = gen(lazy=False)
lazy = gen(lazy=True)

out = "gen_lazycache_r1_alllayers.png"
grid = make_grid((torch.cat([base, lazy]).clamp(-1, 1) + 1) / 2.0, nrow=8, padding=2)
save_image(grid, out)
print(f"saved -> {out}  (上排 baseline / 下排 lazy r=1)")
