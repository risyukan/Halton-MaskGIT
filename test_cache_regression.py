"""回归测试: 既有 cache 方案在引入 LazyMAR Token Cache 后必须逐位不变。

从 git 里取出参照版本的 Network/transformer.py, 用同一份随机权重、同一组 env
配置各跑一次完整采样, 逐位比较 logits 与最终图像。

用法:
    python test_cache_regression.py                # 与 HEAD 比
    python test_cache_regression.py <git-ref>      # 与指定提交比 (改动已提交后用
                                                   # 引入本方案那次提交的父提交)
"""
import contextlib
import importlib.util
import os
import subprocess
import sys
import tempfile
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Network.transformer import Transformer as NewT
from test_lazy_token_cache import DEVICE, StubAE, env, run, assert_bitwise

CFG = dict(input_size=8, nclass=10, hidden_dim=64, codebook_size=64,
           depth=6, heads=4, mlp_dim=256, dropout=0., register=1, proj=1)

GATE = dict(HALTON_PARTIAL_START_LAYER=1, HALTON_PARTIAL_END_LAYER=4)
MODES = {
    "baseline (无 cache)":        {},
    "partial_update (ffn cache)": dict(HALTON_PARTIAL_UPDATE=1, **GATE),
    "ffn cache + refresh N=2":    dict(HALTON_PARTIAL_UPDATE=1, HALTON_CACHE_REFRESH_N=2, **GATE),
    "attn cache":                 dict(HALTON_PARTIAL_UPDATE=1, HALTON_ATTN_CACHE=1, **GATE),
    "layer cache":                dict(HALTON_PARTIAL_UPDATE=1, HALTON_LAYER_CACHE=1, **GATE),
    "layer cache + refresh N=2":  dict(HALTON_PARTIAL_UPDATE=1, HALTON_LAYER_CACHE=1,
                                       HALTON_CACHE_REFRESH_N=2, **GATE),
}


def load_reference(ref):
    """把 <ref> 版本的 transformer.py 作为独立模块加载。"""
    repo = os.path.dirname(os.path.abspath(__file__))
    src = subprocess.check_output(["git", "-C", repo, "show", f"{ref}:Network/transformer.py"])
    path = os.path.join(tempfile.mkdtemp(), "reference_transformer.py")
    with open(path, "wb") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("reference_transformer", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def trainer_for(vit):
    t = types.SimpleNamespace()
    t.vit, t.input_size, t.ae = vit, CFG["input_size"], StubAE()
    t.autocast = contextlib.nullcontext()
    t.args = types.SimpleNamespace(device=DEVICE, mask_value=CFG["codebook_size"],
                                   codebook_size=CFG["codebook_size"])
    return t


def main():
    ref = sys.argv[1] if len(sys.argv) > 1 else "HEAD"
    old_mod = load_reference(ref)
    print(f"参照版本: {ref}\ndevice = {DEVICE}\n")

    torch.manual_seed(0)
    old = old_mod.Transformer(**CFG)
    # Transformer.initialize_weights 把每个 Block 的 AdaLN 调制层零初始化 (DiT 风格),
    # 于是随机模型的 alpha1/alpha2 全为 0, x = x + alpha*(...) 退化成恒等映射 ——
    # 那样整个 transformer 什么都不做, 任何 cache 对比都会平凡通过。必须重新随机化。
    for blk in old.transformer.layers:
        torch.nn.init.normal_(blk.mlp[1].weight, std=0.02)
        torch.nn.init.normal_(blk.mlp[1].bias, std=0.5)
    old = old.to(DEVICE).eval()
    torch.manual_seed(0)
    new = NewT(**CFG).to(DEVICE).eval()
    new.load_state_dict(old.state_dict(), strict=True)   # 同一份 (非零门) 权重
    gates = min(b.mlp[1].weight.abs().max().item() for b in new.transformer.layers)
    assert gates > 0, "AdaLN 门为零 → transformer 是恒等映射, 对比无意义"

    # 参照版本的 forward 可能还没有 cfg_pair 形参, 而打过补丁的 sampler 会传它。
    # 纯桥接, 不改变被测逻辑。
    if "cfg_pair" not in old.forward.__code__.co_varnames:
        _of = old.forward
        old.forward = lambda *a, cfg_pair=False, **kw: _of(*a, **kw)

    fail = 0
    for name, e in MODES.items():
        outs = []
        for vit in (old, new):
            with env(**e):
                outs.append(run(trainer_for(vit), nb_sample=3, steps=10, cfg_w=1.5, seed=42))
        try:
            assert_bitwise(outs[0][2], outs[1][2], "logits")
            assert torch.equal(outs[0][0], outs[1][0]), "image 不一致"
            print(f"  PASS  {name}")
        except Exception as ex:
            fail += 1
            print(f"  FAIL  {name}: {ex}")

    print(f"\n{len(MODES) - fail}/{len(MODES)} 既有方案逐位不变")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
