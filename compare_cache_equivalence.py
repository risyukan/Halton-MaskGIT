"""比较 verify_cache_equivalence.py 生成的两份参考结果。

    python compare_cache_equivalence.py A.pt B.pt

逐构型报告图像的 max/mean 绝对差与 code 不一致数量。实现层面的重构应当得到
全构型 BIT-EXACT; 出现任何 DIFF 都意味着数值行为发生了变化。
"""
import sys
import torch

a = torch.load(sys.argv[1])
b = torch.load(sys.argv[2])
assert a["seed"] == b["seed"] and a["dtype"] == b["dtype"], "seed/dtype 不一致"

print(f"{'config':18s} {'max|dimg|':>12s} {'mean|dimg|':>12s} {'code mismatch':>14s}  verdict")
ok_all = True
for k in a["data"]:
    ia, ib = a["data"][k]["img"], b["data"][k]["img"]
    ca, cb = a["data"][k]["codes"], b["data"][k]["codes"]
    dimg = (ia - ib).abs()
    nmis = int((ca != cb).sum())
    exact = (dimg.max().item() == 0.0) and nmis == 0
    ok_all &= exact
    print(f"{k:18s} {dimg.max().item():12.3e} {dimg.mean().item():12.3e} "
          f"{nmis:8d}/{ca.numel():<6d}  {'BIT-EXACT' if exact else 'DIFF'}")

print("\n=> " + ("全構成でビット単位一致" if ok_all else "差分あり (上を参照)"))
