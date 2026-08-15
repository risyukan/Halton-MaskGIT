# Transformer Encoder architecture
# some part have been borrowed from:
#   - NanoGPT: https://github.com/karpathy/nanoGPT
#   - DiT: https://github.com/facebookresearch/DiT

import math
import os

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange


def param_count(archi, model):
    print(f"Size of model {archi}: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 10 ** 6:.3f}M")


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ── partial-update の active token 抽出/書き戻し ─────────────────────────────
# 以前は bool マスクによる advanced indexing (x[mask] / out[mask] = ...) と
# int(mask[0].sum().item()) を各 Block / 各 Attention で呼んでいた。どちらも内部で
# nonzero() ないし D2H コピーを伴い、CUDA ストリームを毎レイヤ同期させるため、
# CPU が GPU より先行できず kernel launch レイテンシが全て露出していた
# (24 層 × 32 step × CFG 2 経路 で 1 生成あたり数千回の同期)。
#
# 代わりに「昇順の index テンソル (b, k)」を forward の先頭で 1 度だけ作り、
# 以降は gather / scatter_ だけで済ませる。index は昇順なので結果は bool マスク
# indexing とビット単位で同一 (nonzero() も行優先の昇順を返すため)。
def active_idx_from_mask(mask, n_active=None):
    """(b, n) bool → (b, k) int64 の昇順 index。

    n_active (= 各行の True 数, 全行同一) が CPU 側で既知なら同期は一切起きない。
    未知の場合のみ 1 回だけ .sum() で取得する (呼び出しは forward あたり 1 回)。
    """
    if n_active is None:
        n_active = int(mask[0].sum())          # ← ここだけ同期 (forward 1 回につき 1 度)
    # stable な降順 argsort: True(1) が前に集まり、同値内では元の順序が保たれるので
    # 先頭 k 個がそのまま昇順の active index になる。
    return mask.to(torch.uint8).argsort(dim=1, descending=True, stable=True)[:, :n_active]


def gather_active(x, idx):
    """(b, n, d) から idx (b, k) の位置を集める → (b, k, d)。x[mask].view(b,k,d) と同値。"""
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.size(-1)))


def scatter_active(out, idx, src):
    """out の idx 位置を src (b, k, d) で上書き (in-place)。out[mask] = src と同値。

    bool マスク代入 (index_put_) は dtype を暗黙変換するが scatter_ は一致を要求する。
    autocast 下では src が bf16 / out が fp32 になりうるので、ここで揃えておく。
    """
    if src.dtype != out.dtype:
        src = src.to(out.dtype)
    return out.scatter_(1, idx.unsqueeze(-1).expand(-1, -1, out.size(-1)), src)


class FeedForward(nn.Module):
    def __init__(self, dim, h_dim, multiple_of=256, bias=False, dropout=0.):
        super().__init__()
        self.dropout = dropout
        # swinGLU
        h_dim = int(2 * h_dim / 3)
        # make sure it is a power of 256
        h_dim = multiple_of * ((h_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, h_dim, bias=bias)
        self.w2 = nn.Linear(h_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, h_dim, bias=bias)

    def forward(self, x):
        # SwiGLU activation
        x = F.silu(self.w1(x)) * self.w3(x)
        if self.dropout > 0. and self.training:
            x = F.dropout(x, self.dropout)
        return self.w2(x)


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim, linear=False, bias=False)
        self.key_norm = RMSNorm(dim, linear=False, bias=False)

    def forward(self, q, k, v):
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., use_flash=True, bias=False):
        super().__init__()
        self.flash = use_flash
        self.n_local_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.wq = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)
        self.wk = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)
        self.wv = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)
        self.wo = nn.Linear(num_heads * self.head_dim, embed_dim, bias=bias)
        self.qk_norm = QKNorm(num_heads * self.head_dim)
        self.cache = None
        # 上一步本 attn 算出的完整 (pre-gate) 注意力输出 delta, 形状 (b, h_w, d)。
        # 仅在 HALTON_ATTN_CACHE=1 时使用: partial 步里 inactive 位置用它代替 0
        # (与 Block.cached_ffn_delta 对称)。每个 full 步刷新, 跨 generation 由
        # Block.clear_ffn_cache 一并清空。
        self.cached_attn_delta = None

    def forward(self, x, mask=None, active_idx=None):
        """
        active_idx: (b, k) int64 — when provided, Q is computed only for those
        active (newly-released) positions; K/V use all positions.
        Inactive positions receive a zero attention delta so the residual stream
        is not updated via attention.  Every row must hold the same number of
        indices (guaranteed by HaltonSampler's uniform step schedule).
        """
        b, h_w, _ = x.shape

        if active_idx is None:
            # ── Full update (original behaviour) ──────────────────────────
            xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
            # normalize queries and keys
            xq, xk = self.qk_norm(xq, xk, xv)
            xq = xq.view(b, h_w, self.n_local_heads, self.head_dim)
            xk = xk.view(b, h_w, self.n_local_heads, self.head_dim)
            xv = xv.view(b, h_w, self.n_local_heads, self.head_dim)

            # make heads be a batch dim
            xq, xk, xv = (x.transpose(1, 2) for x in (xq, xk, xv))
            # attention
            if self.flash:
                if mask is not None:
                    mask = mask.view(b, 1, 1, h_w)
                output = F.scaled_dot_product_attention(
                    xq, xk, xv, mask,
                    dropout_p=self.dropout if self.training else 0.
                )
            else:
                scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
                if mask is not None:
                    scores = scores + mask  # (bs, heads, seqlen, cache_len + seqlen)
                scores = F.softmax(scores.float(), dim=-1).type_as(xq)
                output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)
            # concatenate all the heads
            output = output.transpose(1, 2).contiguous().view(b, h_w, -1)
            # output projection
            proj = self.wo(output)
            if self.dropout > 0. and self.training:
                proj = F.dropout(proj, self.dropout)
            # 刷新 attn-delta 缓存: 让随后的 partial 步 / refresh 步在 inactive
            # 位置有一份全 token 的基线可回退 (对称于 cached_ffn_delta)。
            if os.environ.get("HALTON_ATTN_CACHE", "0") == "1":
                self.cached_attn_delta = proj.detach()
            return proj

        else:
            # ── Q-only-active: Q from U_t tokens, K/V from all tokens ─────
            # active_idx: (b, n_active) int64, 昇順 (行ごとの個数は一定)
            n_active = active_idx.size(1)
            x_active = gather_active(x, active_idx)           # (b, n_active, d)

            xq = self.wq(x_active)   # (b, n_active, d)
            xk = self.wk(x)          # (b, h_w,      d)
            xv = self.wv(x)          # (b, h_w,      d)

            # QK norm applied independently — different seq lengths are fine
            xq = self.qk_norm.query_norm(xq).to(xv)
            xk = self.qk_norm.key_norm(xk).to(xv)

            xq = xq.view(b, n_active, self.n_local_heads, self.head_dim).transpose(1, 2)
            xk = xk.view(b, h_w,      self.n_local_heads, self.head_dim).transpose(1, 2)
            xv = xv.view(b, h_w,      self.n_local_heads, self.head_dim).transpose(1, 2)

            # Cross-length attention: Q(n_active) attends to KV(h_w)
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                dropout_p=self.dropout if self.training else 0.
            )  # (b, heads, n_active, head_dim)

            output = output.transpose(1, 2).contiguous().view(b, n_active, -1)
            proj = self.wo(output)   # (b, n_active, d)
            if self.dropout > 0. and self.training:
                proj = F.dropout(proj, self.dropout)

            # Inactive 位置的填充: (HALTON_ATTN_CACHE 方案)
            #   有可用缓存 → 取上一步本层的完整 attn delta (与 cached_ffn_delta 对称);
            #   否则 → 退化为 0 (即原始行为, attn 不更新 inactive 残差)。
            if (
                self.cached_attn_delta is not None
                and self.cached_attn_delta.shape == (b, h_w, proj.shape[-1])
                and self.cached_attn_delta.dtype == x.dtype
            ):
                out_full = self.cached_attn_delta.clone()
            else:
                out_full = torch.zeros(b, h_w, proj.shape[-1], device=x.device, dtype=x.dtype)
            scatter_active(out_full, active_idx, proj)
            # 存回合并后的 delta, 供下一步 inactive 位置复用。
            self.cached_attn_delta = out_full.detach()
            return out_full

    def forward_active(self, x, active_idx, mask=None):
        """Layer-output-cache 方案专用的 attention。

        与上面 forward 的 active 分支不同:
          - Q 只对 active token 计算, K/V 对全部 token 计算 (inactive 的 K/V 来自
            上一 step 缓存的本层输入, 已经承载在 x 里);
          - 只返回 active token 的 attention 输出 (b, n_active, d), 不做 inactive
            位置的填充, 也不触碰 cached_attn_delta (那是 HALTON_ATTN_CACHE 方案)。
        active_idx: (b, n_active) int64, 昇順。
        """
        b, h_w, _ = x.shape
        n_active = active_idx.size(1)
        x_active = gather_active(x, active_idx)           # (b, n_active, d)

        xq = self.wq(x_active)   # (b, n_active, d)
        xk = self.wk(x)          # (b, h_w,      d)
        xv = self.wv(x)          # (b, h_w,      d)

        xq = self.qk_norm.query_norm(xq).to(xv)
        xk = self.qk_norm.key_norm(xk).to(xv)

        xq = xq.view(b, n_active, self.n_local_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(b, h_w,      self.n_local_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(b, h_w,      self.n_local_heads, self.head_dim).transpose(1, 2)

        attn_mask = mask.view(b, 1, 1, h_w) if mask is not None else None
        output = F.scaled_dot_product_attention(
            xq, xk, xv, attn_mask,
            dropout_p=self.dropout if self.training else 0.
        )  # (b, heads, n_active, head_dim)

        output = output.transpose(1, 2).contiguous().view(b, n_active, -1)
        proj = self.wo(output)   # (b, n_active, d)
        if self.dropout > 0. and self.training:
            proj = F.dropout(proj, self.dropout)
        return proj


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, linear=True, bias=True):
        super().__init__()
        self.eps = eps
        self.linear = linear
        self.add_bias = bias
        if self.linear:
            self.weight = nn.Parameter(torch.ones(dim))
        if self.add_bias:
            self.bias = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.linear:
            output = self.weight * output
        if self.add_bias:
            output = output + self.bias
        return output


class AdaNorm(nn.Module):
    def __init__(self, x_dim, y_dim):
        super().__init__()
        self.norm_final = RMSNorm(x_dim, linear=True, bias=True, eps=1e-5)
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(y_dim, x_dim * 2))

    def forward(self, x, y):
        shift, scale = self.mlp(y).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return x


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.ln1 = RMSNorm(dim, linear=True, bias=False, eps=1e-5)
        self.attn = Attention(dim, heads, dropout=dropout)
        self.ln2 = RMSNorm(dim, linear=True, bias=False, eps=1e-5)
        self.ff = FeedForward(dim, mlp_dim, dropout=dropout)
        # 上一步本 layer 计算出的 (gated) FFN delta, 用于在 inactive 位置上代替 0。
        # 形状 (b, h_w, d); shape mismatch 时自动重置。
        self.cached_ffn_delta = None
        # 上一步本 layer 的完整输出 (residual stream), 形状 (b, h_w, d)。
        # 仅在 HALTON_LAYER_CACHE=1 时使用: partial 步里 inactive 位置直接沿用它,
        # active 位置用本步重算的输出覆盖 (与 cached_ffn_delta 的 delta 缓存不同,
        # 这里缓存的是整层输出本身)。
        self.cached_layer_output = None

    def clear_ffn_cache(self):
        """采样新一轮生成前调用, 避免跨 generation 串台。
        同时清空 attention-delta 缓存 (HALTON_ATTN_CACHE 模式) 与 layer-output
        缓存 (HALTON_LAYER_CACHE 模式): 采样器只调用 clear_ffn_cache 这一个钩子,
        故三类缓存都在这里一起清。"""
        self.cached_ffn_delta = None
        self.attn.cached_attn_delta = None
        self.cached_layer_output = None

    def forward(self, x, cond, mask=None, active_idx=None):
        """
        active_idx: (b, k) int64 昇順 (Transformer.forward で 1 度だけ構築) —
        当前配置:
          - Attention: 当 active_idx 不为 None 时进入 active-only 模式
                       (Q 仅取 active 位置, K/V 取全部 token; 输出仅写回 active 位置)。
          - FFN: 当 active_idx 不为 None 时只对 active 位置算 FFN;
                  inactive 位置不算 FFN, 改为加上上一步同层缓存的 FFN delta
                  (无缓存时退化为 0)。每次 forward 都把本步的完整 (gated)
                  delta 存回 self.cached_ffn_delta 供下一步使用。
        """
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.mlp(cond).chunk(6, dim=1)

        # ── Layer-output-cache 方案 (HALTON_LAYER_CACHE=1) ──────────────────
        # 与 ffn/attn cache 互斥: 开启后 partial 步走这里, 上面两类 cache 不参与。
        # 机制: 缓存整层输出, 下一 step 只重算 active token
        #   (Q 只取 active, K/V 取全部 —— inactive 的 K/V 来自 x 里承载的上一步
        #    本层输入), FFN 也只算 active; inactive 位置直接沿用缓存的整层输出,
        #   最后把缓存里 active 位置更新为本步新算的输出。
        if active_idx is not None and os.environ.get("HALTON_LAYER_CACHE", "0") == "1":
            return self._forward_layer_cache(
                x, active_idx, mask,
                gamma1, beta1, alpha1, gamma2, beta2, alpha2,
            )

        # Attention: 仅在 HALTON_ATTN_CACHE=1 时进入 active-only + cached-delta 模式
        # (active token 的 Q 对全 token K/V 做 attention, inactive 用上一步缓存);
        # 默认 (开关关闭) 保持 full update —— baseline 完全不变。
        attn_active = (
            active_idx
            if (active_idx is not None
                and os.environ.get("HALTON_ATTN_CACHE", "0") == "1")
            else None
        )
        x = x + alpha1.unsqueeze(1) * self.attn(
            modulate(self.ln1(x), gamma1, beta1),
            mask=mask,
            active_idx=attn_active,
        )
        # FFN: active-only when active_idx is provided, with cached-delta fill-in
        if active_idx is None:
            # full-token FFN; 同时刷新缓存
            ff_delta = alpha2.unsqueeze(1) * self.ff(modulate(self.ln2(x), gamma2, beta2))
            x = x + ff_delta
            self.cached_ffn_delta = ff_delta.detach()
        else:
            b, h_w, d = x.shape
            x_active = gather_active(x, active_idx)                        # (b, n_active, d)
            ff_out = self.ff(modulate(self.ln2(x_active), gamma2, beta2))  # (b, n_active, d)
            active_delta = (alpha2.unsqueeze(1) * ff_out)                  # (b, n_active, d)

            # 起点: inactive 位置取上一步缓存; 没有缓存或形状不匹配则用 0
            if (
                self.cached_ffn_delta is not None
                and self.cached_ffn_delta.shape == x.shape
                and self.cached_ffn_delta.dtype == x.dtype
            ):
                delta = self.cached_ffn_delta.clone()
            else:
                delta = torch.zeros_like(x)
            # 覆盖 active 位置为本步新算的 delta
            scatter_active(delta, active_idx, active_delta)

            x = x + delta
            self.cached_ffn_delta = delta.detach()

        # full-update 步 (active_idx is None) 顺带刷新 layer-output 缓存, 让随后的
        # partial 步 / refresh 步在 inactive 位置有一份全 token 的整层输出可沿用。
        if active_idx is None and os.environ.get("HALTON_LAYER_CACHE", "0") == "1":
            self.cached_layer_output = x.detach()
        return x

    def _forward_layer_cache(self, x, active_idx, mask,
                             gamma1, beta1, alpha1, gamma2, beta2, alpha2):
        """HALTON_LAYER_CACHE 方案的 partial forward。

        输入 x: active 位置承载本步重算的上一层输出, inactive 位置承载上一步缓存
        的上一层输出 (由上层 _forward_layer_cache 构造)。K/V 用整个 x, 因此 inactive
        的 K/V 天然来自缓存。
        """
        b, h_w, d = x.shape

        # ── Attention: Q 仅 active, K/V 全部; 只拿 active 的注意力输出 ──
        h_norm = modulate(self.ln1(x), gamma1, beta1)                  # (b, h_w, d) 供 K/V
        attn_active = self.attn.forward_active(h_norm, active_idx, mask)   # (b, n_active, d)

        # active 位置做残差; inactive 位置不动 (稍后整块用缓存覆盖)
        x_active = gather_active(x, active_idx)                        # (b, n_active, d)
        x_active = x_active + alpha1.unsqueeze(1) * attn_active

        # ── FFN 只算 active token ──
        ff_out = self.ff(modulate(self.ln2(x_active), gamma2, beta2))  # (b, n_active, d)
        x_active = x_active + alpha2.unsqueeze(1) * ff_out             # 本步本层的新输出

        # ── 拼整层输出: inactive 用缓存, active 用新算的; 再写回缓存 ──
        cache_ok = (
            self.cached_layer_output is not None
            and self.cached_layer_output.shape == x.shape
            and self.cached_layer_output.dtype == x.dtype
        )
        # 既定は「缓存张量を in-place 更新してそのまま本層出力として返す」。
        # clone を省くことで、每层每 partial 步一次の (b, n, d) 分配 + 全量拷贝が
        # 消える (large-384/bs8 で約 4 ms/img — fp32 で理論値 4.2 ms/img とほぼ一致)。
        # 数値は clone 版とビット単位で同一 (compare_cache_equivalence.py で確認)。
        #
        # 安全性: 返り値 C_i は次の層に x として渡るだけで、どの経路も x を in-place
        # 変更しない (x = x + delta は新テンソルを作る)。層 i は C_i に書く前に
        # x = C_{i-1} を読み終えており、C_i と C_{i-1} は別物なので read-after-write
        # のハザードもない。
        # ただし grad が有効だと detach() が storage を共有したまま次 step で書き
        # 潰され、保存済み activation を壊す。サンプリングは no_grad なので通常は
        # 該当しないが、念のため grad 有効時は clone にフォールバックする。
        # HALTON_LAYER_CACHE_CLONE=1 で明示的に clone 版へ戻せる (A/B 用)。
        use_inplace = (
            cache_ok
            and not torch.is_grad_enabled()
            and os.environ.get("HALTON_LAYER_CACHE_CLONE", "0") != "1"
        )
        if use_inplace:
            out = self.cached_layer_output
        elif cache_ok:
            out = self.cached_layer_output.clone()
        else:
            # 首个 partial 步还没缓存: inactive 退化为沿用当前输入 x (即上一层此步的
            # 输出), 相当于本层对 inactive 不更新。
            out = x.clone()
        scatter_active(out, active_idx, x_active)
        self.cached_layer_output = out.detach()
        return out


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(Block(dim, heads, mlp_dim, dropout=dropout))

    def clear_ffn_cache(self):
        for blk in self.layers:
            blk.clear_ffn_cache()

    def forward(self, x, cond, mask=None, active_idx=None,
                partial_update_start_layer=3, partial_update_end_layer=21):
        # Layer-level gating (from analyze_ffn_delta_stability):
        #   exclude layer 0 (no prior context) and very top layers — use
        #   partial_update only in the stable mid-stack: start ≤ i ≤ end.
        # Env overrides let small/base runs preserve the "top-K layers full"
        # tail without touching the large defaults.
        _s = os.environ.get("HALTON_PARTIAL_START_LAYER")
        if _s:
            try:
                partial_update_start_layer = int(_s)
            except ValueError:
                pass
        _e = os.environ.get("HALTON_PARTIAL_END_LAYER")
        if _e:
            try:
                partial_update_end_layer = int(_e)
            except ValueError:
                pass
        for i, block in enumerate(self.layers):
            use_partial = partial_update_start_layer <= i <= partial_update_end_layer #use_partial为True时，表示在第3到第21层之间使用partial_update，即FFN只更新active_mask指定的位置；否则在其他层使用full update，即FFN更新所有位置。
            x = block(x, cond, mask=mask, active_idx=active_idx if use_partial else None) #active_idxはTransformerEncoderの引数で、Blockのforwardに渡される。use_partialがTrueのとき、active_idxがBlockのforwardに渡され、FFNはactive_idxで指定された位置のみを更新する。use_partialがFalseのとき、active_idxはNoneとしてBlockのforwardに渡され、FFNは全ての位置を更新する。
        return x


class Transformer(nn.Module):
    """ DiT-like transformer with adaLayerNorm with zero initializations """

    def clear_ffn_cache(self):
        """转发到 TransformerEncoder, 清空每层 Block 的 FFN delta 缓存。"""
        self.transformer.clear_ffn_cache()

    def __init__(self, input_size=16, hidden_dim=768, codebook_size=1024,
                 depth=12, heads=16, mlp_dim=3072, dropout=0., nclass=1000,
                 register=1, proj=1, **kwargs):
        super().__init__()

        self.nclass = nclass
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.codebook_size = codebook_size
        self.proj = proj

        self.cls_emb = nn.Embedding(nclass + 1, hidden_dim)
        self.tok_emb = nn.Embedding(codebook_size + 1, hidden_dim)
        self.pos_emb = nn.Embedding(input_size ** 2, hidden_dim)

        if self.proj > 1:
            self.in_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=2, stride=2, bias=False)
            self.out_proj = nn.Conv2d(
                hidden_dim, hidden_dim * 4, kernel_size=1, stride=1, padding=0, bias=False
            ).to(memory_format=torch.channels_last)

        self.transformer = TransformerEncoder(
            dim=hidden_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout
        )

        self.last_norm = AdaNorm(x_dim=hidden_dim, y_dim=hidden_dim)
        self.head = nn.Linear(hidden_dim, codebook_size + 1)
        self.head.weight = self.tok_emb.weight

        self.register = register
        if self.register > 0:
            self.reg_tokens = nn.Embedding(self.register, hidden_dim)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        nn.init.normal_(self.cls_emb.weight, std=0.02)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

        for block in self.transformer.layers:
            nn.init.constant_(block.mlp[1].weight, 0)
            nn.init.constant_(block.mlp[1].bias, 0)

        if self.proj > 1:
            nn.init.xavier_uniform_(self.in_proj.weight)
            nn.init.xavier_uniform_(self.out_proj.weight)

        if self.register > 0:
            nn.init.normal_(self.reg_tokens.weight, std=0.02)

    def forward(self, x, y, drop_label, mask=None, active_mask=None, active_nnz=None):
        """
        active_mask: (b, h, w) bool — newly-active token positions for
        partial-update mode (Q-only-active attention).
        Pass None for the standard full-update forward pass.
        active_nnz: int|None — 各行の active token 数。呼び出し側 (HaltonSampler) が
        CPU 側で既に知っている値を渡すと、内部の index 構築で device 同期が
        一切発生しない。None なら forward あたり 1 回だけ .sum() で求める。
        """
        b, h, w = x.size()
        h0, w0 = h, w   # original spatial dims before any proj
        x = x.reshape(b, h * w)

        y = torch.where(drop_label, torch.full_like(y, self.nclass), y)
        y = self.cls_emb(y)

        pos = torch.arange(0, w * h, dtype=torch.long, device=x.device)
        pos = self.pos_emb(pos)

        x = self.tok_emb(x) + pos

        if self.proj > 1:
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w, b=b, c=self.hidden_dim).contiguous()
            x = self.in_proj(x)
            _, _, h, w = x.shape   # h, w updated to projected resolution
            x = rearrange(x, 'b c h proj_w -> b (h proj_w) c', proj_h=h, proj_w=w, b=b, c=self.hidden_dim).contiguous()

        # Build sequence-level active_mask aligned with current seq length.
        seq_active_mask = None
        if active_mask is not None:
            if self.proj > 1:
                # Pool (b, h0, w0) → (b, h, w) by OR over each proj×proj patch.
                # view(b, h, proj, w, proj) groups pixels by conv-patch correctly
                # because PyTorch row-major layout maps [ph*proj+sh, pw*proj+sw]
                # to indices [ph, sh, pw, sw] under this reshape.
                am = active_mask.view(b, h, self.proj, w, self.proj)
                am = am.any(2).any(3)          # (b, h, w) — active if any sub-token is
                seq_active_mask = am.view(b, h * w)
            else:
                seq_active_mask = active_mask.view(b, h0 * w0)

            if self.register > 0:
                # Register tokens are never in U_t; they always attend fully.
                reg_false = torch.zeros(b, self.register, dtype=torch.bool, device=x.device)
                seq_active_mask = torch.cat([seq_active_mask, reg_false], dim=1)

        # bool マスク → 昇順 index を「forward あたり 1 度だけ」構築する。
        # 以降 24 層はこの index を gather/scatter で使い回すので、レイヤ毎の
        # nonzero()/.item() による同期が消える。proj>1 のときは OR プーリングで
        # 個数が変わりうるので、渡された active_nnz は使わず数え直す。
        seq_active_idx = None
        if seq_active_mask is not None:
            nnz = active_nnz if (active_nnz is not None and self.proj == 1) else None
            seq_active_idx = active_idx_from_mask(seq_active_mask, nnz)

        if self.register > 0:
            reg = torch.arange(0, self.register, dtype=torch.long, device=x.device)
            x = torch.cat([x, self.reg_tokens(reg).expand(b, self.register, self.hidden_dim)], dim=1)

        x = self.transformer(x, y, mask=mask, active_idx=seq_active_idx)

        x = x[:, :h * w].contiguous()   # drop register tokens

        if self.proj > 1:
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w, b=b, c=self.hidden_dim).contiguous()
            x = self.out_proj(x)
            x = rearrange(
                x, 'b (c s1 s2) h w -> b (h s1 w s2) c',
                s1=self.proj, s2=self.proj, b=b, h=h, w=w, c=self.hidden_dim
            ).contiguous()

        x = self.last_norm(x, y)
        logit = self.head(x)

        return logit


if __name__ == "__main__":
    from thop import profile

    for size in ["tiny", "small", "base"]:
        print(size)
        if size == "tiny":
            hidden_dim, depth, heads = 384, 6, 6
        elif size == "small":
            hidden_dim, depth, heads = 512, 12, 6
        elif size == "base":
            hidden_dim, depth, heads = 768, 12, 12
        elif size == "large":
            hidden_dim, depth, heads = 1024, 24, 16
        elif size == "xlarge":
            hidden_dim, depth, heads = 1152, 28, 16
        else:
            hidden_dim, depth, heads = 768, 12, 12

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_size = 16
        model = Transformer(
            input_size=input_size, nclass=1000, hidden_dim=hidden_dim, codebook_size=16834,
            depth=depth, heads=heads, mlp_dim=hidden_dim * 4, dropout=0.1
        ).to(device)
        code = torch.randint(0, 16384, size=(1, input_size, input_size)).to(device)
        cls = torch.randint(0, 1000, size=(1,)).to(device)
        d_label = (torch.rand(1) < 0.1).to(device)

        flops, params = profile(model, inputs=(code, cls, d_label))
        print(f"FLOPs: {flops//1e9:.2f}G, Params: {params/1e6:.2f}M")
