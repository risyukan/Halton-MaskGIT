"""HALTON_LAZY_VSIM —— LazyMAR 选点方式 (纯 V 相似度 + 随 step 衰减的比例) 的测试。

与 HALTON_LAZY_CACHE 原有的 forced-only / 打分模式的区别是本质的:
本模式 **没有强制集合** —— U_t / U_{t-1} / register 都不再无条件重算, 名额
完全由"入口层 V 相对上一 decoding step 的余弦变化量"排序决定, 名额大小 =
ceil(rho_t * N), rho_t 取自 LazyMAR 的 RETAIN_RATIO_SCHEDULE 并按生成进度重采样。

断言:
  V1  预算 = ceil(rho_t * N), 且 rho_t 随 step 单调不增 (对齐 LazyMAR 的表)
  V2  选点确实是"V 余弦变化量的 TopK", 与独立重算的排序一致
  V3  没有强制集合 —— U_t ∪ U_{t-1} 会有落选 (反证: 原 r=1 模式必全选)
  V4  register token 不再被强制 active, 且各行可以选出不同数量的图像 token
      (head 的 clamp 路径必须扛得住)
  V5  CFG 两半选同一批 token
  V6  rho_t = 1 的 step 全部重算 (等价于全量步), 且不做打分
  V7  step gate 与 lazy cache 完全一致: partial 步的集合逐步相同
  V8  接口保护: 未开 HALTON_LAZY_CACHE / start=0 时明确报错
  V9  跨 generation 缓存清理生效 (两次 generation 逐位一致)

用法:  python test_lazy_vsim.py
"""
import os
import sys
import math

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import Network.transformer as T
from test_lazy_token_cache import (env, make_trainer, run, check, RESULTS,
                                   assert_bitwise, DEVICE)

VSIM_ENV = dict(HALTON_PARTIAL_UPDATE=1, HALTON_LAZY_CACHE=1, HALTON_LAZY_VSIM=1,
                HALTON_LAZY_START_LAYER=1, HALTON_LAZY_END_LAYER=5)


def spy_vsim(trainer, steps=12, nb_sample=4, cfg_w=1.5, seed=1234, **extra_env):
    """跑一次采样, 记录每个 partial 步的 (budget, active_idx, xv, prev_v, cfg_pair)。"""
    seen = []
    orig = T.TransformerEncoder._lazy_select_vsim

    def spy(self, xv, prev_v, budget, cfg_pair):
        idx = orig(self, xv, prev_v, budget, cfg_pair)
        seen.append(dict(budget=budget, idx=idx.detach().clone(),
                         xv=xv.detach().clone(),
                         prev_v=None if prev_v is None else prev_v.detach().clone(),
                         cfg_pair=cfg_pair, n=xv.shape[2]))
        return idx

    T.TransformerEncoder._lazy_select_vsim = spy
    try:
        cfg = dict(VSIM_ENV)
        cfg.update(extra_env)
        with env(**cfg):
            out = run(trainer, nb_sample=nb_sample, steps=steps, cfg_w=cfg_w, seed=seed)
    finally:
        T.TransformerEncoder._lazy_select_vsim = orig
    return out, seen


# ---------------------------------------------------------------------------
def v1_budget_follows_decaying_schedule():
    """预算必须逐步等于 ceil(rho_t * N), rho_t = LazyMAR 表按进度重采样的值。"""
    trainer = make_trainer(seed=3, input_size=8, register=1, depth=6)
    steps = 12
    seen = []                       # (step, budget, n)
    orig = T.TransformerEncoder._forward_lazy

    def spy(self, x, cond, mask, forced_mask, start, end, cfg_pair, step=None,
            total_steps=None, **kw):
        out = orig(self, x, cond, mask, forced_mask, start, end, cfg_pair,
                   step=step, total_steps=total_steps, **kw)
        if forced_mask is not None:
            seen.append((step, total_steps, int(self.lazy_last_active_idx.size(1)),
                         x.size(1)))
        return out

    T.TransformerEncoder._forward_lazy = spy
    try:
        with env(**VSIM_ENV):
            run(trainer, nb_sample=4, steps=steps, cfg_w=1.5)
    finally:
        T.TransformerEncoder._forward_lazy = orig

    assert seen, "没有 partial 步"
    rows = []
    for step, tot, k, n in seen:
        assert tot == steps, (tot, steps)
        rho = T._lazy_vsim_ratio(step, tot)
        want = max(1, min(n, math.ceil(rho * n)))
        assert k == want, f"step {step}: budget {k} != ceil({rho}*{n})={want}"
        rows.append((step, rho, k))
    rhos = [r for _, r, _ in rows]
    assert all(a >= b for a, b in zip(rhos, rhos[1:])), f"rho 没有单调不增: {rhos}"
    assert rhos[0] > rhos[-1], f"rho 全程没衰减: {rhos}"

    tab = [T._lazy_vsim_ratio(t, steps) for t in range(steps)]
    assert tab == sorted(tab, reverse=True) and tab[0] == 1.0 and tab[-1] == 0.05, tab
    # 32 步 (真实配置) 下的表, 顺便钉住 LazyMAR 的映射
    tab32 = [T._lazy_vsim_ratio(t, 32) for t in range(32)]
    assert tab32[:10] == [1.0] * 10 and tab32[10] == 0.6 and tab32[20:] == [0.05] * 12, tab32
    return f"(step, rho, k) = {rows}"


def v2_selection_is_topk_of_v_cosine():
    trainer = make_trainer(seed=5, input_size=8, depth=6)
    _, seen = spy_vsim(trainer, steps=12)
    checked = 0
    for s in seen:
        if s["prev_v"] is None or s["budget"] >= s["n"]:
            continue
        cos = torch.nn.functional.cosine_similarity(
            s["xv"].float(), s["prev_v"].float(), dim=-1).mean(dim=1)
        score = 1.0 - cos
        b = score.size(0)
        if s["cfg_pair"] and b % 2 == 0:              # 与实现一致: 两半共享分数
            half = b // 2
            shared = 0.5 * (score[:half] + score[half:])
            score = torch.cat([shared, shared], dim=0)
        want, _ = torch.sort(score.topk(s["budget"], dim=1).indices, dim=1)
        assert torch.equal(want, s["idx"]), "选点不是 V 余弦变化量的 TopK"
        # 选中集合的分数必须全部 >= 未选中集合的最大分数
        sel = torch.zeros_like(score, dtype=torch.bool).scatter_(1, s["idx"], True)
        assert (score.masked_fill(~sel, float("inf")).min(dim=1).values
                >= score.masked_fill(sel, float("-inf")).max(dim=1).values - 1e-6).all()
        checked += 1
    assert checked, "没有可验证的打分步"
    return f"{checked} 个 partial 步的选点 = TopK(1-cos(V_t, V_(t-1)))"


def v3_no_forced_set():
    """U_t ∪ U_{t-1} 不再被无条件保留 —— 这正是本接口与 lazy cache 的分界。"""
    trainer = make_trainer(seed=7, input_size=8, depth=6)
    (_, _, _, l_U), seen = spy_vsim(trainer, steps=12)
    n_img = 8 * 8
    misses = 0
    total = 0
    # 只看真正做了打分 (budget < n) 的步; 它们与 l_U 的 step 下标由 gate 决定,
    # 这里保守地统计"任一 partial 步是否漏掉过某个刚发布的 token"。
    U_flat = [u.view(u.size(0), -1).bool().to(DEVICE) for u in l_U]
    for s in seen:
        if s["budget"] >= s["n"]:
            continue
        sel = torch.zeros(s["idx"].size(0), s["n"], dtype=torch.bool, device=s["idx"].device)
        sel.scatter_(1, s["idx"], True)
        sel_img = sel[:, :n_img]
        b = sel_img.size(0)
        for U in U_flat:
            u = U.repeat(b // U.size(0), 1) if b != U.size(0) else U
            if int((u & ~sel_img).sum()) > 0:
                misses += 1
                break
        total += 1
    assert total, "没有打分步"
    assert misses > 0, ("每个打分步都覆盖了全部 U_t —— 说明强制集合没被真正去掉 "
                        "(或测试规模太小)")
    return f"{misses}/{total} 个打分步存在落选的 U_t token (无强制集合)"


def v4_register_not_forced_and_rows_may_differ():
    trainer = make_trainer(seed=9, input_size=8, depth=6, register=1)
    _, seen = spy_vsim(trainer, steps=12)
    reg_dropped = 0
    row_diff = 0
    for s in seen:
        if s["budget"] >= s["n"]:
            continue
        n_img = s["n"] - 1                       # register=1
        img_cnt = (s["idx"] < n_img).sum(dim=1)  # 每行选中的图像 token 数
        if int(img_cnt.min()) != int(img_cnt.max()):
            row_diff += 1
        if int(img_cnt.max()) == s["budget"]:    # 该行没选 register
            reg_dropped += 1
    assert reg_dropped > 0, "register token 每步都被选中 → 仍然是强制的?"
    return (f"{reg_dropped} 个步里 register 落选; {row_diff} 个步各行图像 token 数不同 "
            f"(head clamp 路径已覆盖)")


def v5_cfg_halves_share_selection():
    trainer = make_trainer(seed=11, input_size=8, depth=6)
    _, seen = spy_vsim(trainer, steps=12, nb_sample=4, cfg_w=1.5)
    n = 0
    for s in seen:
        if not s["cfg_pair"] or s["idx"].size(0) % 2:
            continue
        h = s["idx"].size(0) // 2
        assert torch.equal(s["idx"][:h], s["idx"][h:]), "CFG 两半选点不同"
        n += 1
    assert n, "没有 cfg_pair 步"
    return f"{n} 个 partial 步两半选择一致"


def v6_ratio_one_steps_recompute_everything():
    """rho_t = 1 的步不该进打分函数 (走便宜的全量路径)。"""
    trainer = make_trainer(seed=13, input_size=8, depth=6)
    _, seen = spy_vsim(trainer, steps=12)
    assert all(s["budget"] < s["n"] for s in seen), \
        "rho=1 的步仍然调用了打分函数"
    # 常数 rho=1 → 一次打分都不做, 但 lazy 路径仍要跑起来
    calls = []
    orig = T.TransformerEncoder._lazy_select_vsim

    def spy(self, *a, **k):
        calls.append(1)
        return orig(self, *a, **k)

    T.TransformerEncoder._lazy_select_vsim = spy
    try:
        cfg = dict(VSIM_ENV); cfg["HALTON_LAZY_VSIM_SCHED"] = "1.0"
        with env(**cfg):
            run(trainer, nb_sample=2, steps=12, cfg_w=1.5)
    finally:
        T.TransformerEncoder._lazy_select_vsim = orig
    assert not calls, f"rho=1 仍打分 {len(calls)} 次"
    st = trainer.vit.get_lazy_stats()
    assert st["vsim_calls"] > 0, "vsim 路径没被走到"
    return f"rho=1: 打分 0 次, vsim_calls={st['vsim_calls']}"


def v7_step_gate_matches_lazy_cache():
    """partial 步落在哪些 step 上, 必须与 HALTON_LAZY_CACHE 完全一致 ——
    本接口只改"选哪些 token", 不改"哪些 step 走 partial"。"""
    trainer = make_trainer(seed=17, input_size=8, depth=6)

    def gate(**extra):
        seen = []
        orig = T.TransformerEncoder._forward_lazy

        def spy(self, x, cond, mask, forced_mask, start, end, cfg_pair, **kw):
            seen.append(forced_mask is not None)
            return orig(self, x, cond, mask, forced_mask, start, end, cfg_pair, **kw)

        T.TransformerEncoder._forward_lazy = spy
        try:
            cfg = dict(HALTON_PARTIAL_UPDATE=1, HALTON_LAZY_CACHE=1,
                       HALTON_LAZY_START_LAYER=1, HALTON_LAZY_END_LAYER=5,
                       HALTON_CACHE_REFRESH_N=2)
            cfg.update(extra)
            with env(**cfg):
                run(trainer, nb_sample=2, steps=12, cfg_w=1.5)
        finally:
            T.TransformerEncoder._forward_lazy = orig
        return seen

    g_cache = gate()
    g_vsim = gate(HALTON_LAZY_VSIM=1)
    assert g_cache == g_vsim, f"step gate 不一致\n cache={g_cache}\n vsim ={g_vsim}"
    return f"REFRESH_N=2 / 12 步: partial 步序列一致 {[int(v) for v in g_vsim]}"


def v8_interface_guards():
    trainer = make_trainer(seed=19, input_size=8, depth=6)
    hits = []
    for cfg, want in [
        (dict(HALTON_PARTIAL_UPDATE=1, HALTON_LAZY_VSIM=1), "HALTON_LAZY_CACHE=1"),
        (dict(HALTON_PARTIAL_UPDATE=1, HALTON_LAZY_CACHE=1, HALTON_LAZY_VSIM=1,
              HALTON_LAZY_START_LAYER=0), "START_LAYER >= 1"),
    ]:
        try:
            with env(**cfg):
                run(trainer, nb_sample=2, steps=8, cfg_w=1.5)
        except RuntimeError as e:
            assert want.split()[0].split("=")[0] in str(e) or "START_LAYER" in str(e), e
            hits.append(want)
        else:
            raise AssertionError(f"未拦截: {cfg}")
    return " / ".join(hits)


def v9_cache_cleared_between_generations():
    trainer = make_trainer(seed=21, input_size=8, depth=6)
    with env(**VSIM_ENV):
        a = run(trainer, nb_sample=2, steps=12, cfg_w=1.5, seed=777)
        b = run(trainer, nb_sample=2, steps=12, cfg_w=1.5, seed=777)
    assert_bitwise(a[2], b[2], "跨 generation 的 logits")
    assert torch.equal(a[0], b[0]), "跨 generation 的 image 不一致"
    return "两次 generation 逐位一致"


def main():
    print(f"device = {DEVICE}, torch = {torch.__version__}\n")
    print("HALTON_LAZY_VSIM (LazyMAR 选点复刻) 测试")
    check("V1 预算 = ceil(rho_t*N) 且随 step 衰减",     v1_budget_follows_decaying_schedule)
    check("V2 选点 = V 余弦变化量的 TopK",             v2_selection_is_topk_of_v_cosine)
    check("V3 没有强制集合 (U_t 会落选)",              v3_no_forced_set)
    check("V4 register 不强制 / 各行图像 token 数可不同", v4_register_not_forced_and_rows_may_differ)
    check("V5 CFG 两半共享同一选择",                   v5_cfg_halves_share_selection)
    check("V6 rho=1 的步不打分, 全部重算",             v6_ratio_one_steps_recompute_everything)
    check("V7 step gate 与 lazy cache 一致",           v7_step_gate_matches_lazy_cache)
    check("V8 接口保护 (LAZY_CACHE / start>=1)",       v8_interface_guards)
    check("V9 跨 generation 缓存清理",                 v9_cache_cleared_between_generations)

    n_fail = sum(1 for ok, _, _ in RESULTS if ok is False)
    n_skip = sum(1 for ok, _, _ in RESULTS if ok is None)
    print(f"\n{len(RESULTS) - n_fail - n_skip}/{len(RESULTS)} 通过"
          + (f", {n_skip} 跳过" if n_skip else "")
          + (f", {n_fail} 失败" if n_fail else ""))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
