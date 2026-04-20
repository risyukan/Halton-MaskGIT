# Transformer Encoder architecture
# some part have been borrowed from:
#   - NanoGPT: https://github.com/karpathy/nanoGPT
#   - DiT: https://github.com/facebookresearch/DiT

import math

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange
from Sampler.halton_sampler import HaltonSampler  # Replace with the correct module if different



def param_count(archi, model):
    print(f"Size of model {archi}: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 10 ** 6:.3f}M")


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# ========= Halton-ToMe helpers =========
def _halton_centers_and_assignments_sq(h_or_w: int, num_centers: int, device):
    """
    使用 HaltonSampler.build_halton_mask(h) 生成 [N,2] (y,x) 样本点，
    前 num_centers 个作为中心，其余 token 依据 2D 欧氏距离分配到最近中心。
    仅支持正方形网格 (h==w) 的场景；N = h*h。
    返回:
      centers_yx: [M,2]
      cluster_ids: [N] 中每个 token 的簇 id (0..M-1)
      counts: [M] 每个簇的成员数
    """
    N = h_or_w * h_or_w
    # 总的 token 数
    M = max(1, int(round(N * 1.0)))  # 默认先占位，实际由外层控制 keep_ratio 决定
    # 这里仅生成 Halton mask，真正的 M 在外部生效
    # 确保至少是 1，避免出现 M = 0 的情况
    halton_mask = HaltonSampler.build_halton_mask(h_or_w).to(device)  # [N,2] (y,x)

    def _build_with_M(M_): # M_ 保留的中心数
        centers_yx = halton_mask[:M_].to(device)               # [M,2]
        # 取前 M 个作为中心
        yy, xx = torch.meshgrid(
            torch.arange(h_or_w, device=device),
            torch.arange(h_or_w, device=device),
            indexing="ij"
        )
        coords = torch.stack([yy, xx], dim=-1).reshape(-1, 2).float()  # [N,2]
        # 计算 N×M 距离并找最近中心
        dist = torch.cdist(coords, centers_yx.float(), p=2)             # [N,M]
        # torch.cdist 计算两两之间的距离。
        # 得到一个 [N, M] 的矩阵：第 i 行表示第 i 个 token 到所有 M 个中心的距离。
        cluster_ids = dist.argmin(dim=1)                                 # [N]
        # cluster_ids 的形状是 [N]，每个元素是 0..M_-1 之间的整数，代表 token 属于哪个簇
        counts = torch.bincount(cluster_ids, minlength=M_).clamp_min(1) # [M]
        # counts 的形状是 [M]，每个元素表示对应簇的成员数量，确保最少为 1
        return centers_yx, cluster_ids, counts
        # centers_yx：中心坐标（形状 [M, 2]，比如 (y, x)）。
        # cluster_ids：每个 token 的簇 ID（形状 [N]）。
        # counts：每个簇的 token 数（形状 [M]，不会小于 1）。

    return _build_with_M
    # 用法：build = _halton_centers_and_assignments_sq(h, None, device); centers, ids, cnt = build(M)
    

def _merge_tokens_spatial(x_spatial, cluster_ids, counts):
    """
    均值合并同簇 token。
    x_spatial: [B,N,C]; cluster_ids: [N]; counts: [M]
    返回 xm: [B,M,C]
    """
    B, N, C = x_spatial.shape
    M = counts.numel()
    xm = x_spatial.new_zeros(B, M, C)
    idx = cluster_ids.view(1, N, 1).expand(B, N, C)           # [B,N,C]
    xm.scatter_add_(1, idx, x_spatial)                        # 同簇求和
    # xm: [B,M,C]，每个中心的特征是其簇内所有 token 特征的和
    xm = xm / counts.view(1, M, 1)                            # 均值
    # 得到均值特征
    return xm


def _unmerge_tokens_spatial(xm_spatial, cluster_ids):
    """
    反合并：把中心特征复制回各成员。
    xm_spatial: [B,M,C]; cluster_ids: [N]
    返回 x: [B,N,C]
    """
    B, M, C = xm_spatial.shape
    N = cluster_ids.numel()
    idx = cluster_ids.view(1, N, 1).expand(B, N, C)           # [B,N,C] 值域 0..M-1
    x = xm_spatial.gather(dim=1, index=idx)                   # 复制中心到成员
    return x
# ======================================

RETAIN_RATIO_SCHEDULE = [
    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,  # 0-7   force_fresh
    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.90,  # 8-15  15开始lazy
    0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 1.00,  # 16-23 23=force_fresh
    0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.10   # 24-31 1.3bei jiasu
]

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
        self.flash = use_flash # use flash attention?
        self.n_local_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.wq = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)
        self.wk = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)
        self.wv = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)
        self.wo = nn.Linear(num_heads * self.head_dim, embed_dim, bias=bias)

        self.qk_norm = QKNorm(num_heads * self.head_dim)

        # will be KVCache object managed by inference context manager
        self.cache = None

    def forward(self, x, mask=None, current=None, cache_dic=None, layer_idx=None):
        b, n, _ = x.shape

        # qkv
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq, xk = self.qk_norm(xq, xk, xv)

        xq = xq.view(b, n, self.n_local_heads, self.head_dim)
        xk = xk.view(b, n, self.n_local_heads, self.head_dim)
        xv = xv.view(b, n, self.n_local_heads, self.head_dim)

        xq, xk, xv = (t.transpose(1, 2) for t in (xq, xk, xv))  # [B, H, N, D]

        # ----------------------------
        # Lazy decoder mode
        # ----------------------------
        if current is not None and current.get("lazy_mar", False):
            layer_cache = cache_dic[layer_idx]

            if current.get("is_force_fresh", False) or ("k" not in layer_cache):
                # full refresh
                layer_cache["k"] = xk.clone()
                layer_cache["v"] = xv.clone()
                k_full = layer_cache["k"]
                v_full = layer_cache["v"]
            else:
                # save previous V before selective update at pruning layer
                if layer_idx == 3 and "v" in layer_cache:
                    current["pre_cache_v"] = layer_cache["v"].clone()

                if torch.all(current["update_mask"]):
                    # full-token update
                    layer_cache["k"] = xk.clone()
                    layer_cache["v"] = xv.clone()
                    k_full = layer_cache["k"]
                    v_full = layer_cache["v"]
                else:
                    # partial update: scatter new kv into full cache
                    k_full = layer_cache["k"].clone()
                    v_full = layer_cache["v"].clone()

                    mask_kv = current["update_mask"].unsqueeze(1).unsqueeze(-1) \
                        .expand(-1, self.n_local_heads, -1, self.head_dim)

                    k_full.masked_scatter_(mask_kv, xk)
                    v_full.masked_scatter_(mask_kv, xv)

                    layer_cache["k"] = k_full
                    layer_cache["v"] = v_full

            # attention: partial q attends full kv cache
            if self.flash:
                # lazy mode 下最好先不用 mask，避免裁剪后 mask 长度不一致
                output = F.scaled_dot_product_attention(
                    xq, k_full, v_full,
                    attn_mask=None,
                    dropout_p=self.dropout if self.training else 0.
                )
            else:
                scores = torch.matmul(xq, k_full.transpose(2, 3)) / math.sqrt(self.head_dim)
                scores = F.softmax(scores.float(), dim=-1).type_as(xq)
                output = torch.matmul(scores, v_full)

            output = output.transpose(1, 2).contiguous().view(b, n, -1)
            proj = self.wo(output)
            if self.dropout > 0. and self.training:
                proj = F.dropout(proj, self.dropout)
            return proj

        # ----------------------------
        # original path
        # ----------------------------
        if self.flash:
            if mask is not None:
                mask = mask.view(b, 1, 1, n)
            output = F.scaled_dot_product_attention(
                xq, xk, xv, mask,
                dropout_p=self.dropout if self.training else 0.
            )
        else:
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(scores, xv)

        output = output.transpose(1, 2).contiguous().view(b, n, -1)
        proj = self.wo(output)
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

    def _prune_tokens(self, x, current, cache_dic, layer_idx):
        """
        x: [B, N, C]
        current['update_mask']: [B, N]  (full sequence mask)
        cache_dic[layer_idx]['v']: [B, H, N, D]
        current['pre_cache_v']: [B, H, N, D]
        """
        if "pre_cache_v" not in current:
            return x
        if layer_idx not in cache_dic or "v" not in cache_dic[layer_idx]:
            return x
        b, n, c = x.shape

        # cosine similarity on V
        cos_sim = F.cosine_similarity(
            cache_dic[layer_idx]["v"],
            current["pre_cache_v"],
            dim=-1
        )  # [B, H, N]

        score = cos_sim.mean(dim=1)  # [B, N]

        # force keep current prediction / previous prediction / registers
        if "mask_to_pred_full" in current:
            score[current["mask_to_pred_full"]] = -1e9
        if "prev_mask_to_pred_full" in current:
            score[current["prev_mask_to_pred_full"]] = -1e9
        if "reg_full" in current:
            score[current["reg_full"]] = -1e9

        # sort ascending: low similarity => high change => keep
        _, inds = torch.sort(score, dim=-1, descending=False)

        cur_ratio = RETAIN_RATIO_SCHEDULE[min(current["step"], len(RETAIN_RATIO_SCHEDULE) - 1)]
        must_keep = int(current["mask_to_pred_len"] + current["prev_mask_to_pred_len"] + 15)
        keep_num = max(must_keep, int(score.shape[1] * cur_ratio))
        keep_num = min(keep_num, n)

        inds = inds[:, :keep_num]

        next_mask = torch.zeros((b, n), device=x.device, dtype=torch.bool)
        next_mask.scatter_(1, inds, True)

        current["origi_update_mask"] = current["update_mask"]
        current["update_mask"] = next_mask
        current["next_mask"] = next_mask

        x = torch.masked_select(
            x,
            next_mask.unsqueeze(-1).expand(-1, -1, c)
        ).reshape(b, -1, c)

        return x

    def _unprune_tokens(self, x, current):
        """
        x: [B, N_keep, C]
        restore to [B, N_full, C]
        """
        b, _, c = x.shape
        n_full = current["next_mask"].shape[1]

        full_x = torch.zeros((b, n_full, c), device=x.device, dtype=x.dtype)
        full_x.masked_scatter_(
            current["next_mask"].unsqueeze(-1).expand(-1, -1, c),
            x
        )

        current["update_mask"] = current["origi_update_mask"]
        current["origi_update_mask"] = None
        current["next_mask"] = None
        return full_x

    def forward(self, x, cond, mask=None, current=None, cache_dic=None, layer_idx=None, depth=None):
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.mlp(cond).chunk(6, dim=1)

        x = x + alpha1.unsqueeze(1) * self.attn(
            modulate(self.ln1(x), gamma1, beta1),
            mask=mask,
            current=current,
            cache_dic=cache_dic,
            layer_idx=layer_idx
        )

        # pruning only in lazy mode, at layer 3
        if current is not None and current.get("lazy_mar", False):
            if (
                (not current.get("is_force_fresh", False))
                and layer_idx == 3
                and "pre_cache_v" in current
                and layer_idx in cache_dic
                and "v" in cache_dic[layer_idx]
            ):
                x = self._prune_tokens(x, current, cache_dic, layer_idx)

        x = x + alpha2.unsqueeze(1) * self.ff(modulate(self.ln2(x), gamma2, beta2))

        # restore full sequence at final layer
        if current is not None and current.get("lazy_mar", False):
            if (not current.get("is_force_fresh", False)) and layer_idx == depth - 1 and current.get("next_mask", None) is not None:
                x = self._unprune_tokens(x, current)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(Block(dim, heads, mlp_dim, dropout=dropout))

    def forward(self, x, cond, mask=None, current=None, cache_dic=None):
        depth = len(self.layers)
        for i, block in enumerate(self.layers):
            x = block(
                x, cond, mask=mask,
                current=current,
                cache_dic=cache_dic,
                layer_idx=i,
                depth=depth
            )
        return x


class Transformer(nn.Module):
    """ DiT-like transformer with adaLayerNorm with zero initializations """
    def __init__(self, input_size=16, hidden_dim=768, codebook_size=1024,
                 depth=12, heads=16, mlp_dim=3072, dropout=0., nclass=1000,
                 register=1, proj=1, **kwargs):
        super().__init__()

        self.nclass = nclass                                             # Number of classes
        self.input_size = input_size                                     # Number of tokens as input
        self.hidden_dim = hidden_dim                                     # Hidden dimension of the transformer
        self.codebook_size = codebook_size                               # Amount of code in the codebook
        self.proj = proj                                                 # Projection

        self.cls_emb = nn.Embedding(nclass + 1, hidden_dim)              # Embedding layer for the class token
        self.tok_emb = nn.Embedding(codebook_size + 1, hidden_dim)       # Embedding layer for the 'visual' token
        self.pos_emb = nn.Embedding(input_size ** 2, hidden_dim)         # Learnable Positional Embedding

        if self.proj > 1:
            self.in_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=2, stride=2, bias=False)
            self.out_proj = nn.Conv2d(
                hidden_dim, hidden_dim*4, kernel_size=1, stride=1, padding=0, bias=False
            ).to(memory_format=torch.channels_last)

        # The Transformer Encoder a la BERT :)
        self.transformer = TransformerEncoder(dim=hidden_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout)

        self.last_norm = AdaNorm(x_dim=hidden_dim, y_dim=hidden_dim)   # Last Norm

        self.head = nn.Linear(hidden_dim, codebook_size + 1)
        self.head.weight = self.tok_emb.weight  # weight tied with the tok_emb layer

        self.register = register
        if self.register > 0:
            self.reg_tokens = nn.Embedding(self.register, hidden_dim)

        self.initialize_weights()  # Init weight
        # ---- Halton-Token-Merge 配置 ----
        self.tome_keep_ratio = kwargs.get("tome_keep_ratio", 1.0)  # 1.0=不合并；比如 0.5=保留一半 token
        self.tome_merge_layer_idx = kwargs.get("tome_merge_layer_idx", 0)  # 第 1 层前合并
        self.tome_unmerge_before_idx = kwargs.get("tome_unmerge_before_idx", -1)  # 最后一层前反合并
        self.tome_random_roll = kwargs.get("tome_random_roll", False)  # 如需随机 roll Halton 序列可用
        self._tome_cache = {}  # key=(h,w,M,device) -> {"centers":..., "ids":..., "counts":...}

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Init embedding
        nn.init.normal_(self.cls_emb.weight, std=0.02)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

        # Zero-out adaNorm modulation layers in blocks:
        for block in self.transformer.layers:
            nn.init.constant_(block.mlp[1].weight, 0)
            nn.init.constant_(block.mlp[1].bias, 0)

        # Init proj layer
        if self.proj > 1:
            nn.init.xavier_uniform_(self.in_proj.weight)
            nn.init.xavier_uniform_(self.out_proj.weight)

        # Init embedding
        if self.register > 0:
            nn.init.normal_(self.reg_tokens.weight, std=0.02)

    def forward(self, x, y, drop_label, mask=None, lazy_state=None):
      # x是code，y是class label，drop_label是bool值
        # global_masked_token形状是 [batch, input_size, input_size] 的布尔张量，
        # 表示当前整幅图里哪些 token 还没被预测（全局未解码的位置）
        #
        # current_mask 形状是 [nb_sample, input_size, input_size]，
        # True 代表这一轮要更新的token。
    

        b, h, w = x.size()
        x = x.reshape(b, h*w)

        # Drop the label if drop_label
        y = torch.where(drop_label, torch.full_like(y, self.nclass), y)
        y = self.cls_emb(y)

        pos = torch.arange(0, w*h, dtype=torch.long, device=x.device)
        pos = self.pos_emb(pos)

        x = self.tok_emb(x) + pos

        # reshape, proj to smaller space, reshape (patchify!)
        if self.proj > 1:
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w, b=b, c=self.hidden_dim).contiguous()
            x = self.in_proj(x)
            _, _, h, w = x.shape
            x = rearrange(x, 'b c h proj_w -> b (h proj_w) c', proj_h=h, proj_w=w, b=b, c=self.hidden_dim).contiguous()

        if self.register > 0:
            reg = torch.arange(0, self.register, dtype=torch.long, device=x.device)
            x = torch.cat([x, self.reg_tokens(reg).expand(b, self.register, self.hidden_dim)], dim=1)

        current = None
        cache_dic = None

        if lazy_state is not None and lazy_state.get("lazy_mar", False):
            bsz, seq_len, _ = x.shape

            # init per-layer cache
            if "cache" not in lazy_state:
                lazy_state["cache"] = {i: {} for i in range(len(self.transformer.layers))}
            cache_dic = lazy_state["cache"]

            # full sequence mask
            update_mask = torch.ones((bsz, seq_len), device=x.device, dtype=torch.bool)

            # image-token level masks (before register concat)
            mask_to_pred = lazy_state.get("mask_to_pred", torch.zeros((bsz, seq_len), device=x.device, dtype=torch.bool))
            prev_mask_to_pred = lazy_state.get("prev_mask_to_pred", torch.zeros((bsz, seq_len), device=x.device, dtype=torch.bool))

            # if register tokens exist, append False for them
            if self.register > 0:
                reg_full = torch.zeros((bsz, self.register), device=x.device, dtype=torch.bool)
                mask_to_pred_full = torch.cat([mask_to_pred, reg_full], dim=1)
                prev_mask_to_pred_full = torch.cat([prev_mask_to_pred, reg_full], dim=1)

                reg_keep = torch.zeros((bsz, seq_len), device=x.device, dtype=torch.bool)
                reg_keep[:, -self.register:] = True
            else:
                mask_to_pred_full = mask_to_pred
                prev_mask_to_pred_full = prev_mask_to_pred
                reg_keep = torch.zeros((bsz, seq_len), device=x.device, dtype=torch.bool)

            current = {
                "lazy_mar": True,
                "is_force_fresh": lazy_state.get("is_force_fresh", False),
                "step": lazy_state.get("step", 0),
                "update_mask": update_mask,
                "mask_to_pred_full": mask_to_pred_full,
                "prev_mask_to_pred_full": prev_mask_to_pred_full,
                "mask_to_pred_len": int(mask_to_pred_full[0].sum().item()),
                "prev_mask_to_pred_len": int(prev_mask_to_pred_full[0].sum().item()),
                "reg_full": reg_keep,
            }

        x = self.transformer(x, y, mask=mask, current=current, cache_dic=cache_dic)

        # drop the register
        x = x[:, :h*w].contiguous()

        if self.proj > 1:
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w, b=b, c=self.hidden_dim).contiguous()
            x = self.out_proj(x)
            x = rearrange(x, 'b (c s1 s2) h w -> b (h s1 w s2) c', s1=self.proj, s2=self.proj, b=b, h=h, w=w, c=self.hidden_dim).contiguous()

        x = self.last_norm(x, y)
        logit = self.head(x)

        return logit




if __name__ == "__main__":
    from thop import profile
    # 测试不同规模的transformer的FLOPs和参数量
    for size in ["tiny", "small", "base"]: # "large", "xlarge"]:
        # size = "tiny"
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
        model = Transformer(input_size=input_size, nclass=1000, hidden_dim=hidden_dim, codebook_size=16834,
                            depth=depth, heads=heads, mlp_dim=hidden_dim * 4, dropout=0.1).to(device)
        # model = torch.compile(model)
        code = torch.randint(0, 16384, size=(1, input_size, input_size)).to(device)
        cls = torch.randint(0, 1000, size=(1,)).to(device)
        d_label = (torch.rand(1) < 0.1).to(device)

        flops, params = profile(model, inputs=(code, cls, d_label))
        print(f"FLOPs: {flops//1e9:.2f}G, Params: {params/1e6:.2f}M")

