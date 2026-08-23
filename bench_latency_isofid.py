"""iso-FID 对照: layer cache vs lazy token cache 在相同 refresh N 下的实测 latency。

50k FID 已证实两者在同一个 N 下 FID 等价 (2.6953/2.6951 @ N=2,
3.1822/3.1899 @ N=4, 10.7726/10.8181 @ none), 所以同 N 的 latency 之比就是
"等质量下的净加速"。方法与 bench_latency.py 一致。

用法: python bench_latency_isofid.py [batch] [warmup] [timed]
"""
import os, sys, time, statistics, torch
from Utils.utils import load_args_from_file
from Trainer.cls_trainer import MaskGIT
from Sampler.halton_sampler import HaltonSampler

BATCH  = int(sys.argv[1]) if len(sys.argv) > 1 else 16
WARMUP = int(sys.argv[2]) if len(sys.argv) > 2 else 1
TIMED  = int(sys.argv[3]) if len(sys.argv) > 3 else 5

args = load_args_from_file("Config/base_cls2img.yaml")
args.device = torch.device("cuda"); args.vit_size = "large"; args.img_size = 384
args.compile = False; args.dtype = "float32"; args.resume = True
args.vit_folder = f"./saved_networks/ImageNet_{args.img_size}_{args.vit_size}.pth"
model = MaskGIT(args)
sampler = HaltonSampler(sm_temp_min=1, sm_temp_max=1.2, temp_pow=1, temp_warmup=0,
                        w=2, sched_pow=2, step=32, randomize=True, top_k=-1)
bl = [1, 7, 282, 604, 724, 179, 681, 850]
labels = torch.LongTensor([bl[i % len(bl)] for i in range(BATCH)]).to(args.device)


def set_cfg(mode, refresh_n):
    for k in ("HALTON_LAYER_CACHE", "HALTON_LAZY_CACHE", "HALTON_ATTN_CACHE",
              "HALTON_LAZY_START_LAYER", "HALTON_LAZY_END_LAYER",
              "HALTON_PARTIAL_START_LAYER", "HALTON_PARTIAL_END_LAYER"):
        os.environ.pop(k, None)
    os.environ["HALTON_CACHE_REFRESH_N"] = str(refresh_n)
    if mode == "baseline":
        os.environ["HALTON_PARTIAL_UPDATE"] = "0"
    elif mode == "layer":
        os.environ["HALTON_PARTIAL_UPDATE"] = "1"
        os.environ["HALTON_LAYER_CACHE"] = "1"
    elif mode == "lazy":
        os.environ["HALTON_PARTIAL_UPDATE"] = "1"
        os.environ["HALTON_LAZY_CACHE"] = "1"
        os.environ["HALTON_LAZY_CACHE_RATIO"] = "1.0"
        os.environ["HALTON_LAZY_HEAD"] = "1"


def sync(): torch.cuda.synchronize()

def time_sampler():
    sync(); t0 = time.perf_counter()
    _ = sampler(trainer=model, nb_sample=BATCH, labels=labels, verbose=False,
                partial_update=None)[0]
    sync(); return time.perf_counter() - t0

def time_decode():
    isz = model.input_size
    code = torch.randint(0, args.codebook_size, (BATCH, isz, isz), device=args.device)
    sync(); t0 = time.perf_counter()
    with torch.no_grad():
        torch.clamp(model.ae.decode_code(torch.clamp(code, 0, args.codebook_size-1)), -1, 1)
    sync(); return time.perf_counter() - t0

def bench(fn):
    for _ in range(WARMUP): fn()
    ts = [fn() for _ in range(TIMED)]
    return statistics.mean(ts), (statistics.stdev(ts) if len(ts) > 1 else 0.0)

FID = {("layer",2):2.6953, ("lazy",2):2.6951, ("layer",4):3.1822, ("lazy",4):3.1899,
       ("layer",0):10.7726, ("lazy",0):10.8181}
CONFIGS = [("baseline","baseline",0)] + [
    (f"{m} cache N={n if n else 'none'}", m, n)
    for n in (2,4,0) for m in ("layer","lazy")]

print(f"\n== iso-FID latency | large-384 | batch={BATCH} | warmup={WARMUP} timed={TIMED} "
      f"| {torch.cuda.get_device_name(0)} ==\n")
dec, _ = bench(time_decode)
print(f"VQGAN decode: {dec*1000/BATCH:.2f} ms/img (已从 transf 扣除)\n")
print(f"{'config':22s}{'FID(50k)':>10s}{'transf ms/img':>15s}{'speedup':>10s}")
base = None
res = {}
for name, mode, n in CONFIGS:
    set_cfg(mode, n)
    m, _ = bench(time_sampler)
    tr = m - dec
    if base is None: base = tr
    f = FID.get((mode, n))
    res[(mode, n)] = base / tr
    print(f"{name:22s}{(f'{f:.4f}' if f else '2.5410'):>10s}"
          f"{tr*1000/BATCH:15.2f}{base/tr:9.3f}x")
print(f"\n等质量净增益 (lazy / layercache, 同 N 同 FID):")
for n in (2,4,0):
    print(f"  N={n if n else 'none':<5} {res[('lazy',n)]/res[('layer',n)]:.3f}x")
