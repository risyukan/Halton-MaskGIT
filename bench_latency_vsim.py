"""端到端 latency: LazyMAR 选点复刻 (HALTON_LAZY_VSIM) vs baseline / lazy cache。

方法与 bench_latency_refresh.py 完全一致 (同进程只加载一次模型, cuda.synchronize
包夹, warmup 后多次取均值, VQGAN 解码单独计时并从 total 里扣掉), 只是把配置换成:

  baseline                无 cache
  lazycache r=1 N=2       active = U_t ∪ U_{t-1} ∪ register (纯调度选点), 全部 24 层
  lazyvsim  N=2           active = TopK_{rho_t*N}(1 - cos(V_l3(t), V_l3(t-1))),
                          无强制 token, rho_t 随 step 衰减 (LazyMAR 的表), 层 3..23

三档共用同一套采样调度 (step=32, gate 5..30, REFRESH_N=2, cfg=0.5)。

用法:  python bench_latency_vsim.py [batch] [warmup] [timed]
默认    python bench_latency_vsim.py 16 1 5
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

CFG_W      = float(os.environ.get("BENCH_CFG_W", "0.5"))
REFRESH_N  = int(os.environ.get("BENCH_REFRESH_N", "2"))
VSIM_START = int(os.environ.get("BENCH_LAZY_START", "3"))
VSIM_END   = int(os.environ.get("BENCH_LAZY_END", "23"))
VSIM_SCHED = os.environ.get("BENCH_VSIM_SCHED", "lazymar")

args = load_args_from_file("Config/base_cls2img.yaml")
args.device = torch.device("cuda")
args.vit_size = "large"
args.img_size = 384
args.compile = False
args.dtype = "float32"
args.resume = True
args.vit_folder = f"./saved_networks/ImageNet_{args.img_size}_{args.vit_size}.pth"
model = MaskGIT(args)

sampler = HaltonSampler(sm_temp_min=1.0, sm_temp_max=1.0, temp_pow=1, temp_warmup=1,
                        w=CFG_W, sched_pow=2, step=32, randomize=True, top_k=-1)

base_labels = [1, 7, 282, 604, 724, 179, 681, 850]
labels = torch.LongTensor([base_labels[i % len(base_labels)] for i in range(BATCH)]).to(args.device)

ALL_KEYS = ("HALTON_PARTIAL_UPDATE", "HALTON_CACHE_REFRESH_N", "HALTON_LAZY_CACHE",
            "HALTON_LAZY_CACHE_RATIO", "HALTON_LAZY_HEAD", "HALTON_ATTN_CACHE",
            "HALTON_LAYER_CACHE", "HALTON_LAZY_VSIM", "HALTON_LAZY_VSIM_SCHED",
            "HALTON_LAZY_START_LAYER", "HALTON_LAZY_END_LAYER",
            "HALTON_PARTIAL_START_LAYER", "HALTON_PARTIAL_END_LAYER")


def set_cfg(**kw):
    for k in ALL_KEYS:
        os.environ.pop(k, None)
    for k, v in kw.items():
        os.environ[k] = str(v)


CONFIGS = [
    ("baseline", dict(HALTON_PARTIAL_UPDATE=0)),
    (f"lazycache r=1 N={REFRESH_N}", dict(
        HALTON_PARTIAL_UPDATE=1, HALTON_LAZY_CACHE=1, HALTON_LAZY_CACHE_RATIO=1.0,
        HALTON_LAZY_HEAD=1, HALTON_LAZY_START_LAYER=0, HALTON_LAZY_END_LAYER=23,
        HALTON_CACHE_REFRESH_N=REFRESH_N)),
    (f"lazyvsim N={REFRESH_N} L{VSIM_START}-{VSIM_END}", dict(
        HALTON_PARTIAL_UPDATE=1, HALTON_LAZY_CACHE=1, HALTON_LAZY_VSIM=1,
        HALTON_LAZY_VSIM_SCHED=VSIM_SCHED, HALTON_LAZY_HEAD=1,
        HALTON_LAZY_START_LAYER=VSIM_START, HALTON_LAZY_END_LAYER=VSIM_END,
        HALTON_CACHE_REFRESH_N=REFRESH_N)),
]


def sync():
    torch.cuda.synchronize()


def time_sampler():
    sync(); t0 = time.perf_counter()
    _ = sampler(trainer=model, nb_sample=BATCH, labels=labels,
                verbose=False, partial_update=None)[0]
    sync()
    return time.perf_counter() - t0


def time_decode():
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


print(f"\n== latency bench (LazyMAR V-similarity selection) | large-384 | batch={BATCH} | "
      f"warmup={WARMUP} timed={TIMED} | cfg_w={CFG_W} | {torch.cuda.get_device_name(0)} ==\n")

dec_mean, dec_std = bench(time_decode)
print(f"VQGAN decode      : {dec_mean*1000/BATCH:7.2f} ms/img  "
      f"(batch {dec_mean*1000:7.1f}±{dec_std*1000:.1f} ms)\n")

print(f"{'config':28s} {'total ms/img':>13s} {'transf ms/img':>14s} {'±ms':>7s} {'speedup':>9s}")
base_tr = None
for name, cfg in CONFIGS:
    set_cfg(**cfg)
    m, sd = bench(time_sampler)
    tr = m - dec_mean
    if base_tr is None:
        base_tr = tr
    spd = base_tr / tr
    print(f"{name:28s} {m*1000/BATCH:11.2f}   {tr*1000/BATCH:12.2f}   "
          f"{sd*1000/BATCH:6.2f}  {spd:7.3f}x")
    if cfg.get("HALTON_LAZY_CACHE"):
        st = getattr(model.vit, "module", model.vit).get_lazy_stats()
        print(f"{'':28s} └ partial k/N={st['partial_active_ratio']:.4f} "
              f"({st['partial_calls']} 步) / 全量 {st['full_calls']} 步; "
              f"vsim={st['vsim_calls']} scored={st['scored_calls']} "
              f"kv_miss={st['kv_miss']} score_miss={st['score_miss']} "
              f"head(p/f/miss)={st['head_partial_calls']}/{st['head_full_calls']}/{st['head_miss']}")

print("\n注: transf = total - VQGAN decode (各档解码相同); speedup 以 transformer-only 计。")
