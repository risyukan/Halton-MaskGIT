"""干净的采样 latency benchmark (单卡, 固定 batch)。

测五档: baseline / dual N=2 / dual no-refresh / lazy token cache (r=1, 全层),
lazy 再拆成 head 全量 / head 也只算 active 两档 (后者是等价变换, 见 transformer.py)。
- 同一进程只加载一次模型 (env 变量在每次 forward 内读取, 可逐次切换)。
- cuda.synchronize 包夹, warmup 后多次取均值±std。
- VQGAN 解码单独计时 (各档相同), 用 total - decode 得到 transformer-only latency。
- 同时打印理论 FLOPs 比例做对照。

用法:  python bench_latency.py [batch] [warmup] [timed]
默认    python bench_latency.py 16 1 5
"""
import os
import sys
import time
import statistics
import torch

from Utils.utils import load_args_from_file
from Trainer.cls_trainer import MaskGIT
from Sampler.halton_sampler import HaltonSampler

BATCH  = int(sys.argv[1]) if len(sys.argv) > 1 else 16
WARMUP = int(sys.argv[2]) if len(sys.argv) > 2 else 1
TIMED  = int(sys.argv[3]) if len(sys.argv) > 3 else 5

# ---- model (large-384), 只加载一次 ----
args = load_args_from_file("Config/base_cls2img.yaml")
args.device = torch.device("cuda")
args.vit_size = "large"
args.img_size = 384
args.compile = False
args.dtype = "float32"
args.resume = True
args.vit_folder = f"./saved_networks/ImageNet_{args.img_size}_{args.vit_size}.pth"
model = MaskGIT(args)

sampler = HaltonSampler(sm_temp_min=1, sm_temp_max=1.2, temp_pow=1, temp_warmup=0,
                        w=2, sched_pow=2, step=32, randomize=True, top_k=-1)

base_labels = [1, 7, 282, 604, 724, 179, 681, 850]
labels = torch.LongTensor([base_labels[i % len(base_labels)] for i in range(BATCH)]).to(args.device)


def set_cfg(partial, attn, refresh_n, lazy=False, lazy_ratio=1.0, lazy_head=True):
    os.environ["HALTON_PARTIAL_UPDATE"] = "1" if partial else "0"
    os.environ["HALTON_ATTN_CACHE"]     = "1" if attn else "0"
    os.environ["HALTON_CACHE_REFRESH_N"] = str(refresh_n)
    # LazyMAR Token Cache: r=1 -> active = U_t ∪ U_{t-1} ∪ register, 无 V 打分,
    # 层区间留空 -> 默认覆盖全部 24 层。
    os.environ["HALTON_LAZY_CACHE"]       = "1" if lazy else "0"
    os.environ["HALTON_LAZY_CACHE_RATIO"] = str(lazy_ratio)
    os.environ["HALTON_LAZY_HEAD"] = "1" if lazy_head else "0"
    for k in ("HALTON_LAZY_START_LAYER", "HALTON_LAZY_END_LAYER",
              "HALTON_PARTIAL_START_LAYER", "HALTON_PARTIAL_END_LAYER"):
        os.environ.pop(k, None)


def sync():
    torch.cuda.synchronize()


def time_sampler():
    """返回一次完整生成 (transformer 采样 + VQGAN 解码) 的耗时(s)。"""
    sync(); t0 = time.perf_counter()
    _ = sampler(trainer=model, nb_sample=BATCH, labels=labels,
                verbose=False, partial_update=None)[0]
    sync()
    return time.perf_counter() - t0


def time_decode():
    """单独的 VQGAN 解码耗时(s), 各档相同, 用于从 total 中扣除。"""
    isz = model.input_size
    code = torch.randint(0, args.codebook_size, (BATCH, isz, isz), device=args.device)
    sync(); t0 = time.perf_counter()
    with torch.no_grad():
        x = model.ae.decode_code(torch.clamp(code, 0, args.codebook_size - 1))
        x = torch.clamp(x, -1, 1)
    sync()
    return time.perf_counter() - t0


def bench(fn):
    for _ in range(WARMUP):
        fn()
    ts = [fn() for _ in range(TIMED)]
    return statistics.mean(ts), (statistics.stdev(ts) if len(ts) > 1 else 0.0)


# ---- 理论 FLOPs 比例 (large d=1024 s=577; lazy 一档见 flops_lazy_token_cache.py) ----
THEO = {"baseline": 1.000, "dual N=2": 0.744, "dual no-refresh": 0.489,
        "lazy r=1 fullhead": 0.2706, "lazy r=1 +head": 0.2342}

CONFIGS = [
    ("baseline",          dict(partial=False, attn=False, refresh_n=0)),
    ("dual N=2",          dict(partial=True,  attn=True,  refresh_n=2)),
    ("dual no-refresh",   dict(partial=True,  attn=True,  refresh_n=0)),
    ("lazy r=1 fullhead", dict(partial=True,  attn=False, refresh_n=0, lazy=True,
                               lazy_head=False)),
    ("lazy r=1 +head",    dict(partial=True,  attn=False, refresh_n=0, lazy=True,
                               lazy_head=True)),
]

print(f"\n== latency bench | large-384 | batch={BATCH} | warmup={WARMUP} timed={TIMED} | "
      f"{torch.cuda.get_device_name(0)} ==\n")

# decode 基线 (与配置无关)
dec_mean, dec_std = bench(time_decode)
print(f"VQGAN decode      : {dec_mean*1000/BATCH:7.2f} ms/img  "
      f"(batch {dec_mean*1000:7.1f}±{dec_std*1000:.1f} ms)\n")

print(f"{'config':18s} {'total ms/img':>13s} {'transf ms/img':>14s} "
      f"{'img/s(transf)':>13s} {'speedup':>8s} {'theo':>6s}")
base_tr = None
for name, kw in CONFIGS:
    set_cfg(**kw)
    m, sd = bench(time_sampler)
    total_ms = m * 1000 / BATCH
    transf_ms = (m - dec_mean) * 1000 / BATCH          # 扣掉解码
    transf_ips = BATCH / (m - dec_mean)
    if base_tr is None:
        base_tr = m - dec_mean
    spd = base_tr / (m - dec_mean)
    print(f"{name:18s} {total_ms:11.2f}   {transf_ms:12.2f}   "
          f"{transf_ips:11.1f}   {spd:6.3f}x  {1/THEO[name]:5.3f}x")
    if kw.get("lazy"):
        st = getattr(model.vit, "module", model.vit).get_lazy_stats()
        print(f"{'':18s} └ 实测 active: partial 步 k/N={st['partial_active_ratio']:.4f} "
              f"({st['partial_calls']} 步) / 全量步 {st['full_calls']} 步; "
              f"打分={st['scored_calls']} kv_miss={st['kv_miss']} "
              f"head(partial/full/miss)={st['head_partial_calls']}/"
              f"{st['head_full_calls']}/{st['head_miss']}")

print("\n注: transf = total - VQGAN decode (各档解码相同)。speedup 以 transformer-only 计;")
print("    理论列 = 1/FLOPs比例, 见前述 FLOPs 模型。")
