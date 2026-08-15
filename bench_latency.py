"""论文用 latency benchmark (单卡)。

设计要点 (与旧版的区别):
  1. 全部 HALTON_* 环境变量每次显式赋值 —— 包括 HALTON_LAYER_CACHE 与 layer gate。
     旧版从不设置 LAYER_CACHE, 既测不到 layer-cache 方案, 外部 export 还会静默污染
     baseline。
  2. sampler 与 FID 评测 (main.py --test-only -> Trainer/cls_trainer.py) 的构造方式
     一致: sm_temp_min/max, temp_warmup, w=cfg_w, sched_pow, randomize 全部对齐,
     这样 latency 表和 FID 表可以画在同一张 Pareto 图上。
  3. 计时用 CUDA event 分段, 只在一次生成结束后统一 sync 读数, 运行期间零 stall:
        t_total  = 端到端 (perf_counter + 末尾 sync)
        t_vit    = 所有 transformer forward 的 GPU 时间之和
        t_decode = VQGAN decode 的 GPU 时间
        t_other  = total - vit - decode  (采样器 CPU 侧开销: halton mask 构建、
                   Categorical 采样等; 各配置相同, 会稀释相对加速比, 故单列)
     旧版把 decode 单独测一次再相减, 误差不传播, speedup 也没有区间。
  4. round-robin 交错执行 + 每轮轮换顺序, 抵消 GPU 频率/温度漂移带来的系统性偏差
     (旧版固定 baseline 先跑, 后跑的配置被系统性偏袒)。报告 median 与 IQR。
  5. TF32 显式设定并记录 (fp32 下 TF32 开关会让 matmul 差数倍), 同时记录 torch/CUDA/
     driver/GPU/SM 时钟/温度, 以及 peak memory。
  6. 理论 MACs 不再硬编码, 按真实 Halton 调度逐 step 逐 layer 累加 (含 step gate /
     layer gate / refresh 周期 / cfg 双路 / 输出 head)。
  7. batch × dtype × step × method × refresh 的矩阵扫描, 结果直接写 CSV。

用法:
    # 主表 (方法对比)
    python bench_latency.py --methods baseline,ffn,attn,layer --refresh 2,4 \
        --batch 16 --dtype float32 --repeat 10 --out results/latency_main.csv

    # step-reduction 对照 (审稿人必问的 baseline)
    python bench_latency.py --methods baseline --steps 32,24,20,16 --batch 16 \
        --out results/latency_steps.csv

    # 不加载模型, 只看配置与理论 MACs
    python bench_latency.py --dry-run
"""
import argparse
import csv
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time

import torch

# ── 该测哪些 HALTON_* ─────────────────────────────────────────────────────────
HALTON_ENV_KEYS = (
    "HALTON_PARTIAL_UPDATE",
    "HALTON_ATTN_CACHE",
    "HALTON_LAYER_CACHE",
    "HALTON_LAYER_CACHE_CLONE",
    "HALTON_CACHE_REFRESH_N",
    "HALTON_PARTIAL_START_LAYER",
    "HALTON_PARTIAL_END_LAYER",
)

# method -> (partial, attn_cache, layer_cache, default layer gate)
# layer gate 的默认值对齐各自 FID sweep 所用的 launch/ 脚本:
#   ffn / attn : 3..21   (results/halton_large384_{partialupdate,attncache}_sweep.txt)
#   layer      : 0..23   (results/halton_large384_layercache_alllayers_sweep.txt)
METHODS = {
    "baseline": dict(partial=False, attn=False, layer=False, gate=None),
    "ffn":      dict(partial=True,  attn=False, layer=False, gate=(3, 21)),
    "attn":     dict(partial=True,  attn=True,  layer=False, gate=(3, 21)),
    "layer":    dict(partial=True,  attn=False, layer=True,  gate=(0, 23)),
    # layer cache の旧実装 (キャッシュを毎回 clone する版)。既定の layer とは
    # 数値的に完全等価 (compare_cache_equivalence.py で確認済み) で、clone の
    # コストだけが違う。同一プロセス内で round-robin 比較すると <1% でも見える。
    "layer_clone": dict(partial=True, attn=False, layer=True, gate=(0, 23),
                        clone=True),
}

VIT_SIZES = {   # hidden_dim, depth, heads —— 与 Trainer/cls_trainer.transformer_size 一致
    "tiny":   (384, 6, 6),
    "small":  (512, 12, 6),
    "base":   (768, 12, 12),
    "large":  (1024, 24, 16),
    "xlarge": (1152, 28, 16),
}


# ══════════════════════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════════════════════
class Config:
    """一个待测配置。"""

    def __init__(self, method, refresh_n, step, batch, dtype, gate):
        spec = METHODS[method]
        self.method = method
        self.refresh_n = refresh_n if spec["partial"] else 0
        self.step = step
        self.batch = batch
        self.dtype = dtype
        self.partial = spec["partial"]
        self.attn = spec["attn"]
        self.layer = spec["layer"]
        self.force_clone = spec.get("clone", False)
        self.gate = gate if gate is not None else spec["gate"]

    @property
    def name(self):
        if not self.partial:
            return f"baseline/step{self.step}"
        return f"{self.method}_N{self.refresh_n}/step{self.step}"

    @property
    def key(self):
        return (self.method, self.refresh_n, self.step, self.batch, self.dtype)

    def apply_env(self):
        """全部 6 个变量每次都显式写入, 不依赖外部状态。"""
        os.environ["HALTON_PARTIAL_UPDATE"] = "1" if self.partial else "0"
        os.environ["HALTON_ATTN_CACHE"] = "1" if self.attn else "0"
        os.environ["HALTON_LAYER_CACHE"] = "1" if self.layer else "0"
        os.environ["HALTON_LAYER_CACHE_CLONE"] = "1" if self.force_clone else "0"
        os.environ["HALTON_CACHE_REFRESH_N"] = str(self.refresh_n)
        if self.gate is None:
            os.environ.pop("HALTON_PARTIAL_START_LAYER", None)
            os.environ.pop("HALTON_PARTIAL_END_LAYER", None)
        else:
            os.environ["HALTON_PARTIAL_START_LAYER"] = str(self.gate[0])
            os.environ["HALTON_PARTIAL_END_LAYER"] = str(self.gate[1])


# ══════════════════════════════════════════════════════════════════════════════
# 理论 MACs —— 按真实 Halton 调度逐 step / 逐 layer 累加
# ══════════════════════════════════════════════════════════════════════════════
def halton_active_counts(input_size, step):
    """复刻 HaltonSampler 的 r 调度, 返回每个 step 的 |U_t| (与 randomize 无关)。"""
    counts, prev_r = [], 0
    for index in range(step):
        ratio = (index + 1) / step
        r = 1 - (math.acos(ratio) / (math.pi * 0.5))
        r = int(r * (input_size ** 2))
        r = max(index + 1, r)
        counts.append(r - prev_r)
        prev_r = r
    return counts


def layer_macs_full(n, d):
    """全 token 一层的 MACs: attention(4nd^2 + 2n^2 d) + FFN(8nd^2)。"""
    return 12 * n * d * d + 2 * n * n * d


def layer_macs_ffn_active(n, k, d):
    """attention 全量 + FFN 只算 k 个 active token。"""
    return (4 * n * d * d + 2 * n * n * d) + 8 * k * d * d


def layer_macs_qactive(n, k, d):
    """Q 只取 active、K/V 全量 + FFN 只算 active (attn-cache / layer-cache 共用)。"""
    return (2 * k * d * d + 2 * n * d * d + 2 * k * n * d) + 8 * k * d * d


def theoretical_macs(cfg, input_size, register, d, depth, codebook):
    """返回 (总 MACs, 每 step 的 active token 数列表)。"""
    n = input_size ** 2 + register          # 序列长度 (含 register token)
    img_tok = input_size ** 2               # head 只作用在图像 token 上
    u_counts = halton_active_counts(input_size, cfg.step)

    head_macs = img_tok * d * codebook
    adaln_macs = depth * 6 * d * d          # 每层 adaLN 的 cond->6d (per-sample)

    total = 0
    gated_counter = 0
    actives = []
    for t in range(cfg.step):
        k = None
        if cfg.partial and 5 <= t < 31:
            is_refresh = cfg.refresh_n >= 1 and (gated_counter % cfg.refresh_n == 0)
            gated_counter += 1
            if not is_refresh:
                # active = U_{t-1} ∪ U_t, 两者由 Halton 构造保证不相交
                k = u_counts[t] + (u_counts[t - 1] if t > 0 else 0)
        actives.append(k)

        for i in range(depth):
            in_gate = cfg.gate is not None and cfg.gate[0] <= i <= cfg.gate[1]
            if k is None or not in_gate:
                total += layer_macs_full(n, d)
            elif cfg.layer or cfg.attn:
                total += layer_macs_qactive(n, k, d)
            else:
                total += layer_macs_ffn_active(n, k, d)
        total += head_macs + adaln_macs

    return total, actives


# ══════════════════════════════════════════════════════════════════════════════
# 计时
# ══════════════════════════════════════════════════════════════════════════════
class EventTimer:
    """用 CUDA event 记录一个函数的累计 GPU 时间, 运行期间不 sync。"""

    def __init__(self):
        self.pairs = []

    def wrap(self, fn):
        def inner(*a, **kw):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            out = fn(*a, **kw)
            e.record()
            self.pairs.append((s, e))
            return out
        return inner

    def reset(self):
        self.pairs.clear()

    def total_ms(self):
        # 调用前必须已经 synchronize
        return sum(s.elapsed_time(e) for s, e in self.pairs)


def physical_gpu_index():
    """CUDA_VISIBLE_DEVICES を踏まえた物理 GPU 番号 (nvidia-smi -i に渡す用)。

    これを間違えると、実際に走っているのとは別の (アイドルな) GPU のクロックを
    読んでしまい「210 MHz で回っていた」ような無意味な記録になる。
    """
    logical = torch.cuda.current_device() if torch.cuda.is_available() else 0
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not vis:
        return logical
    try:
        return int(vis.split(",")[logical].strip())
    except (IndexError, ValueError):
        return logical


def gpu_state():
    """采样 SM 时钟与温度, 用于事后确认没有降频。"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits", "-i", str(physical_gpu_index())],
            stderr=subprocess.DEVNULL, timeout=5).decode().strip().split(",")
        return dict(sm_clock_mhz=float(out[0]), temp_c=float(out[1]), power_w=float(out[2]))
    except Exception:
        return dict(sm_clock_mhz=float("nan"), temp_c=float("nan"), power_w=float("nan"))


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--methods", default="baseline,ffn,attn,layer",
                   help="逗号分隔: " + ",".join(METHODS))
    p.add_argument("--refresh", default="2",
                   help="逗号分隔的 cache-refresh N (只作用于非 baseline); 0 = no-refresh")
    p.add_argument("--steps", default="32", help="逗号分隔的采样步数")
    p.add_argument("--batch", default="16", help="逗号分隔的 batch size")
    p.add_argument("--dtype", default="float32", help="逗号分隔: float32,bfloat16,float16")
    p.add_argument("--layer-gate", default=None,
                   help="覆盖 layer gate, 形如 '3,21'; 缺省用各方法对齐 FID 的默认值")
    p.add_argument("--vit-size", default="large", choices=list(VIT_SIZES))
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument("--cfg-w", type=float, default=0.5, help="与 FID 评测一致 (large=0.5)")
    p.add_argument("--temp-warmup", type=int, default=1,
                   help="与 launch/eval_fid_*.sh 的 --temp-warmup 一致 (默认 1); "
                        "不影响计算量, 仅为让 latency 表与 FID 表同配置")
    p.add_argument("--warmup", type=int, default=3, help="丢弃的前 N 轮 (每配置各 N 次)")
    p.add_argument("--repeat", type=int, default=10, help="计入统计的轮数")
    p.add_argument("--compile", action="store_true", help="启用 torch.compile")
    p.add_argument("--tf32", action="store_true",
                   help="fp32 下允许 TF32 matmul (默认关闭, 与 FID 评测一致)")
    p.add_argument("--out", default=None, help="CSV 输出路径")
    p.add_argument("--dry-run", action="store_true", help="不加载模型, 只打印配置与理论 MACs")
    return p.parse_args()


def main():
    args = parse_args()

    gate = None
    if args.layer_gate:
        a, b = args.layer_gate.split(",")
        gate = (int(a), int(b))

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    refreshes = [int(x) for x in args.refresh.split(",")]
    steps = [int(x) for x in args.steps.split(",")]
    batches = [int(x) for x in args.batch.split(",")]
    dtypes = [d.strip() for d in args.dtype.split(",") if d.strip()]

    # 配置展开 (baseline 与 refresh 无关, 去重)
    configs = []
    for dt in dtypes:
        for bs in batches:
            for st in steps:
                for m in methods:
                    ns = [0] if not METHODS[m]["partial"] else refreshes
                    for n in ns:
                        configs.append(Config(m, n, st, bs, dt, gate))

    d, depth, heads = VIT_SIZES[args.vit_size]
    input_size = args.img_size // 16      # f_factor=16
    register, codebook = 1, 16384

    # ── 理论 MACs ──
    theo = {}
    for c in configs:
        macs, actives = theoretical_macs(c, input_size, register, d, depth, codebook)
        if args.cfg_w != 0:
            macs *= 2                      # CFG: cond + uncond 两路 forward
        theo[c.key] = macs

    def ref_key(c):
        return ("baseline", 0, c.step, c.batch, c.dtype)

    print(f"\n== 配置 {len(configs)} 个 | vit={args.vit_size} d={d} depth={depth} "
          f"seq={input_size**2 + register} | cfg_w={args.cfg_w} ==\n")
    print(f"{'config':26s} {'gate':>8s} {'theo GMACs/img':>15s} {'theo speedup':>13s}")
    for c in configs:
        rk = theo.get(ref_key(c))
        sp = (rk / theo[c.key]) if rk else float("nan")
        g = f"{c.gate[0]}..{c.gate[1]}" if c.gate else "-"
        print(f"{c.name:26s} {g:>8s} {theo[c.key]/1e9:15.1f} {sp:12.3f}x")

    if args.dry_run:
        return

    # ── 环境固定与记录 ──
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.tf32)
    torch.backends.cudnn.benchmark = True

    from Utils.utils import load_args_from_file
    from Trainer.cls_trainer import MaskGIT
    from Sampler.halton_sampler import HaltonSampler

    margs = load_args_from_file("Config/base_cls2img.yaml")
    margs.device = torch.device("cuda")
    margs.vit_size = args.vit_size
    margs.img_size = args.img_size
    margs.compile = args.compile
    margs.dtype = dtypes[0]
    margs.resume = True
    margs.vit_folder = f"./saved_networks/ImageNet_{args.img_size}_{args.vit_size}.pth"
    model = MaskGIT(margs)
    model.vit.eval()

    env = dict(
        gpu=torch.cuda.get_device_name(0),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        driver=subprocess.getoutput(
            "nvidia-smi --query-gpu=driver_version --format=csv,noheader -i 0").strip(),
        python=platform.python_version(),
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        tf32_cudnn=torch.backends.cudnn.allow_tf32,
        compile=bool(args.compile),
        cfg_w=args.cfg_w,
        vit_size=args.vit_size,
        img_size=args.img_size,
    )
    print("\n环境: " + json.dumps(env, ensure_ascii=False))

    # ── 计时器挂载 (instance 属性覆盖, nn.Module.__call__ 会走到它) ──
    vit_timer, dec_timer = EventTimer(), EventTimer()
    model.vit.forward = vit_timer.wrap(model.vit.forward)
    model.ae.decode_code = dec_timer.wrap(model.ae.decode_code)

    base_labels = [1, 7, 282, 604, 724, 179, 681, 850]
    label_cache = {}

    def labels_for(bs):
        if bs not in label_cache:
            label_cache[bs] = torch.LongTensor(
                [base_labels[i % len(base_labels)] for i in range(bs)]).to(margs.device)
        return label_cache[bs]

    def make_sampler(step):
        """与 Trainer/cls_trainer.py:62 的构造完全一致 (FID 评测走的就是那条路径)。"""
        return HaltonSampler(sm_temp_min=margs.sm_temp_min, sm_temp_max=margs.sm_temp,
                             temp_pow=1, temp_warmup=args.temp_warmup, w=args.cfg_w,
                             sched_pow=margs.sched_pow, step=step,
                             randomize=margs.randomize, top_k=margs.top_k)

    samplers = {st: make_sampler(st) for st in steps}
    for s in samplers.values():          # halton mask 预构建, 不计入计时
        s.basic_halton_mask = s.build_halton_mask(model.input_size)

    # dtype は Trainer/cls_trainer.py:48-51 と同じく autocast だけで決まる (重みは
    # fp32 のまま)。モデルは 1 度しかロードしないので、構成ごとにここを差し替える。
    # これを忘れると複数 dtype 指定時に全構成が最初の dtype で走ってしまう。
    from contextlib import nullcontext
    AUTOCAST = {
        "float32": nullcontext(),
        "bfloat16": torch.amp.autocast("cuda", dtype=torch.bfloat16),
        "float16": torch.amp.autocast("cuda", dtype=torch.float16),
    }
    for _dt in dtypes:
        if _dt not in AUTOCAST:
            raise SystemExit(f"未知の dtype: {_dt} (float32/bfloat16/float16)")

    def run_once(c):
        c.apply_env()
        model.autocast = AUTOCAST[c.dtype]
        vit_timer.reset()
        dec_timer.reset()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with torch.no_grad():
            samplers[c.step](trainer=model, nb_sample=c.batch, labels=labels_for(c.batch),
                             verbose=False, partial_update=None)
        torch.cuda.synchronize()
        total = (time.perf_counter() - t0) * 1000.0
        return dict(total=total, vit=vit_timer.total_ms(), decode=dec_timer.total_ms(),
                    peak_mb=torch.cuda.max_memory_allocated() / 2 ** 20)

    # ── round-robin 执行: 每轮轮换顺序, 抵消频率/温度漂移 ──
    samples = {c.key: [] for c in configs}
    rounds = args.warmup + args.repeat
    clocks = []
    print(f"\n执行 {rounds} 轮 (前 {args.warmup} 轮为 warmup, 丢弃) × {len(configs)} 配置 ...")
    for r in range(rounds):
        order = configs[r % len(configs):] + configs[:r % len(configs)]
        for c in order:
            res = run_once(c)
            if r >= args.warmup:
                samples[c.key].append(res)
        if r >= args.warmup:
            clocks.append(gpu_state())
        print(f"  round {r + 1}/{rounds}{' (warmup)' if r < args.warmup else ''}", flush=True)

    # ── 汇总 ──
    def stat(vals):
        med = statistics.median(vals)
        q = sorted(vals)
        iqr = (q[int(0.75 * (len(q) - 1))] - q[int(0.25 * (len(q) - 1))]) if len(q) > 3 else 0.0
        return med, iqr

    rows = []
    med_total = {}
    med_vit = {}
    for c in configs:
        rs = samples[c.key]
        med_total[c.key] = statistics.median([x["total"] for x in rs])
        med_vit[c.key] = statistics.median([x["vit"] for x in rs])

    for c in configs:
        rs = samples[c.key]
        t_med, t_iqr = stat([x["total"] for x in rs])
        v_med, v_iqr = stat([x["vit"] for x in rs])
        d_med, _ = stat([x["decode"] for x in rs])
        o_med = t_med - v_med - d_med
        rk = ref_key(c)
        rows.append(dict(
            method=c.method, refresh_n=c.refresh_n, step=c.step, batch=c.batch,
            dtype=c.dtype, gate=(f"{c.gate[0]}..{c.gate[1]}" if c.gate else ""),
            total_ms_img=t_med / c.batch, total_iqr_ms_img=t_iqr / c.batch,
            vit_ms_img=v_med / c.batch, vit_iqr_ms_img=v_iqr / c.batch,
            decode_ms_img=d_med / c.batch, other_ms_img=o_med / c.batch,
            img_per_s=c.batch / (t_med / 1000.0),
            speedup_total=(med_total[rk] / t_med) if rk in med_total else float("nan"),
            speedup_vit=(med_vit[rk] / v_med) if rk in med_vit else float("nan"),
            theo_speedup=(theo[rk] / theo[c.key]) if rk in theo else float("nan"),
            theo_gmacs_img=theo[c.key] / 1e9,
            peak_mb=max(x["peak_mb"] for x in rs),
            n=len(rs),
        ))

    print(f"\n{'config':22s} {'dt':>8s} {'bs':>4s} {'total':>9s} {'vit':>9s} "
          f"{'dec':>7s} {'other':>7s} {'img/s':>7s} {'spd_tot':>8s} {'spd_vit':>8s} "
          f"{'theo':>7s} {'peak MB':>8s}")
    for r in rows:
        cname = (f"baseline/s{r['step']}" if r["method"] == "baseline"
                 else f"{r['method']}_N{r['refresh_n']}/s{r['step']}")
        print(f"{cname:22s} {r['dtype']:>8s} {r['batch']:4d} "
              f"{r['total_ms_img']:8.2f}± {r['vit_ms_img']:8.2f} "
              f"{r['decode_ms_img']:7.2f} {r['other_ms_img']:7.2f} {r['img_per_s']:7.2f} "
              f"{r['speedup_total']:7.3f}x {r['speedup_vit']:7.3f}x "
              f"{r['theo_speedup']:6.3f}x {r['peak_mb']:8.0f}")
    print("\n列说明: total/vit/dec/other 均为 ms/img (median); "
          "vit = transformer forward 的 GPU 时间, dec = VQGAN decode,")
    print("other = 采样器 CPU 侧开销 (halton mask 构建 / Categorical 采样), 各配置相同。")
    print("spd_tot = 端到端加速比, spd_vit = transformer-only 加速比, theo = 理论 MACs 比。")
    if clocks:
        sm = [c["sm_clock_mhz"] for c in clocks]
        tp = [c["temp_c"] for c in clocks]
        print(f"\nGPU 状态 (计入统计的各轮): SM clock {min(sm):.0f}-{max(sm):.0f} MHz, "
              f"温度 {min(tp):.0f}-{max(tp):.0f} °C")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["# " + json.dumps(env, ensure_ascii=False)])
            w.writerow([f"# warmup={args.warmup} repeat={args.repeat} "
                        f"round-robin, median over {args.repeat} runs"])
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
