# Transformer Encoder architecture
# some part have been borrowed from:
#   - NanoGPT: https://github.com/karpathy/nanoGPT
#   - DiT: https://github.com/facebookresearch/DiT

import math

import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple, Callable
from einops import rearrange
from Sampler.halton_sampler import HaltonSampler  # Replace with the correct module if different
from Sampler.halton_sampler import _halton_centers_and_assignments_sq
from Sampler.halton_sampler import _merge_tokens_spatial
from Sampler.halton_sampler import _unmerge_tokens_spatial


def param_count(archi, model):
    print(f"Size of model {archi}: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 10 ** 6:.3f}M")


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def do_nothing(x: torch.Tensor, mode:str=None):
    return x


def mps_gather_workaround(input, dim, index):
    if input.shape[-1] == 1:
        return torch.gather(
            input.unsqueeze(-1),
            dim - 1 if dim < 0 else dim,
            index.unsqueeze(-1)
        ).squeeze(-1)
    else:
        return torch.gather(input, dim, index)


def bipartite_soft_matching_random2d(metric: torch.Tensor,
                                     w: int, h: int, sx: int, sy: int, r: int,
                                     no_rand: bool = False,
                                     generator: torch.Generator = None) -> Tuple[Callable, Callable]:
    """
    Partitions the tokens into src and dst and merges r tokens from src to dst.
    Dst tokens are partitioned by choosing one randomy in each (sx, sy) region.

    Args:
     - metric [B, N, C]: metric to use for similarity
     - w: image width in tokens
     - h: image height in tokens
     - sx: stride in the x dimension for dst, must divide w
     - sy: stride in the y dimension for dst, must divide h
     - r: number of tokens to remove (by merging)
     - no_rand: if true, disable randomness (use top left corner only)
     - rand_seed: if no_rand is false, and if not None, sets random seed.
    """
    B, N, _ = metric.shape

    if r <= 0:
        return do_nothing, do_nothing

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather
    
    with torch.no_grad():
        hsy, wsx = h // sy, w // sx

        # For each sy by sx kernel, randomly assign one token to be dst and the rest src
        if no_rand:
            rand_idx = torch.zeros(hsy, wsx, 1, device=metric.device, dtype=torch.int64)
        else:
            rand_idx = torch.randint(sy*sx, size=(hsy, wsx, 1), device=generator.device, generator=generator).to(metric.device)
        
        # The image might not divide sx and sy, so we need to work on a view of the top left if the idx buffer instead
        idx_buffer_view = torch.zeros(hsy, wsx, sy*sx, device=metric.device, dtype=torch.int64)
        idx_buffer_view.scatter_(dim=2, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
        idx_buffer_view = idx_buffer_view.view(hsy, wsx, sy, sx).transpose(1, 2).reshape(hsy * sy, wsx * sx)

        # Image is not divisible by sx or sy so we need to move it into a new buffer
        if (hsy * sy) < h or (wsx * sx) < w:
            idx_buffer = torch.zeros(h, w, device=metric.device, dtype=torch.int64)
            idx_buffer[:(hsy * sy), :(wsx * sx)] = idx_buffer_view
        else:
            idx_buffer = idx_buffer_view

        # We set dst tokens to be -1 and src to be 0, so an argsort gives us dst|src indices
        rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1)

        # We're finished with these
        del idx_buffer, idx_buffer_view

        # rand_idx is currently dst|src, so split them
        num_dst = hsy * wsx
        a_idx = rand_idx[:, num_dst:, :] # src
        b_idx = rand_idx[:, :num_dst, :] # dst

        def split(x):
            C = x.shape[-1]
            src = gather(x, dim=1, index=a_idx.expand(B, N - num_dst, C))
            dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
            return src, dst

        # Cosine similarity between A and B
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)

        # Can't reduce more than the # tokens in src
        r = min(a.shape[1], r)

        # Find the most similar greedily
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        n, t1, c = src.shape
        
        unm = gather(src, dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = gather(src, dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        _, _, c = unm.shape

        src = gather(dst, dim=-2, index=dst_idx.expand(B, r, c))

        # Combine back to the original shape
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(B, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx).expand(B, r, c), src=src)

        return out

    return merge, unmerge


# ===== ToMe minimal helpers =====
class ToMeCfg:
    def __init__(self, keep_ratio=1.0, sx=2, sy=2, reduce="mean", no_rand=False, seed=None):
        self.keep_ratio = keep_ratio  # 1.0 表示不合并，sx 和 sy 表示合并的区域大小，4存1，reduce 表示合并方式
        self.sx = sx
        self.sy = sy
        self.reduce = reduce          # "mean" 或 "sum"
        self.no_rand = no_rand
        self.seed = seed

def tome_apply_once(x, h, w, reg_len, cfg: "ToMeCfg", generator: "torch.Generator"):
    """
    仅对 x 的前 H*W 段（空间 token）做 ToMe 合并。尾部 reg_len 不动。
    返回：x_merged, unmerge_fn, merged_len
    """
    if cfg is None or cfg.keep_ratio >= 1.0: # 不进行token merge的情况
        return x, do_nothing, h * w

    spatial_len = h * w # 空间 token 数量
    x_sp, x_reg = x[:, :spatial_len, :], (x[:, spatial_len:, :] if reg_len > 0 else None) # 分割空间 token 和class token

    N = spatial_len 
    keep = max(1, int(cfg.keep_ratio * N))
    r = max(0, N - keep) # 需要合并的 token 数量
    if r == 0: # 不进行token merge的情况
        return x, do_nothing, spatial_len

    if cfg.seed is not None: # 固定随机种子
        generator.manual_seed(cfg.seed)

    merge, unmerge = bipartite_soft_matching_random2d(
        metric=x_sp, w=w, h=h, sx=cfg.sx, sy=cfg.sy, r=r,
        no_rand=cfg.no_rand, generator=generator
    ) # no_rand表示是否禁用随机性, generator表示随机数生成器
    x_sp = merge(x_sp, mode=cfg.reduce)
    x_merged = torch.cat([x_sp, x_reg], dim=1) if x_reg is not None else x_sp # 拼回 class token
    return x_merged, unmerge, x_sp.shape[1]


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

    def forward(self, x, mask=None):
        b, h_w, _ = x.shape
        # calculate query, key, value and split out heads
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
            output = F.scaled_dot_product_attention(xq, xk, xv, mask, dropout_p=self.dropout if self.training else 0.)
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

    def forward(self, x, cond, mask=None,
                tome_cfg: "ToMeCfg" = None,
                spatial_hw: "Tuple[int,int]" = None,
                reg_len: int = 0,
                tome_generator: "torch.Generator" = None):
        """
        新增参数：
          - tome_cfg: ToMe 配置（None 或 keep_ratio>=1.0 则不启用）
          - spatial_hw: (H, W)，仅对前 H*W token 合并
          - reg_len: 注册 token 数（尾部），不参与合并
          - tome_generator: 随机数发生器
        """
        H, W = spatial_hw if spatial_hw is not None else (None, None)
        gen = tome_generator

        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.mlp(cond).chunk(6, dim=1)

        # === Self-Attention 前合并 ===
        if H is not None and W is not None and tome_cfg is not None and tome_cfg.keep_ratio < 1.0:
            x, unmerge_attn, merged_len_a = tome_apply_once(x, H, W, reg_len, tome_cfg, gen)
        else:
            unmerge_attn = do_nothing
            merged_len_a = x.shape[1] - reg_len  # 空间段长度（若未合并）

        # === Self-Attention ===
        # 最小改动：mask 在合并后长度变化，这里设为 None
        x = x + alpha1.unsqueeze(1) * self.attn(modulate(self.ln1(x), gamma1, beta1), mask=None)

        # === Self-Attention 后反合并 ===
        if H is not None and W is not None and not (unmerge_attn is do_nothing):
            x_sp, x_reg = x[:, :merged_len_a, :], (x[:, merged_len_a:, :] if reg_len > 0 else None)
            x_sp = unmerge_attn(x_sp)
            x = torch.cat([x_sp, x_reg], dim=1) if x_reg is not None else x_sp

        # === MLP 前合并 ===
        if H is not None and W is not None and tome_cfg is not None and tome_cfg.keep_ratio < 1.0:
            x, unmerge_mlp, merged_len_m = tome_apply_once(x, H, W, reg_len, tome_cfg, gen)
        else:
            unmerge_mlp = do_nothing
            merged_len_m = x.shape[1] - reg_len

        # === MLP ===
        x = x + alpha2.unsqueeze(1) * self.ff(modulate(self.ln2(x), gamma2, beta2))

        # === MLP 后反合并 ===
        if H is not None and W is not None and not (unmerge_mlp is do_nothing):
            x_sp, x_reg = x[:, :merged_len_m, :], (x[:, merged_len_m:, :] if reg_len > 0 else None)
            x_sp = unmerge_mlp(x_sp)
            x = torch.cat([x_sp, x_reg], dim=1) if x_reg is not None else x_sp

        return x




class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(Block(dim, heads, mlp_dim, dropout=dropout))

    def forward(self, x, cond, mask=None,
                tome_cfg: "ToMeCfg" = None,
                spatial_hw: "Tuple[int,int]" = None,
                reg_len: int = 0,
                tome_generator: "torch.Generator" = None):
        for block in self.layers:
            x = block(
                x, cond, mask=mask,                 # 注意：在 Block 内部强制 mask=None 以最小改动
                tome_cfg=tome_cfg,
                spatial_hw=spatial_hw,
                reg_len=reg_len,
                tome_generator=tome_generator
            )
        return x

class Transformer(nn.Module):
    """ DiT-like transformer with adaLayerNorm with zero initializations """
    def __init__(self, input_size=16, hidden_dim=768, codebook_size=1024,
                 depth=12, heads=16, mlp_dim=3072, dropout=0., nclass=1000,
                 register=1, proj=1,
                 # ===== 新增 ToMe 参数（可默认不启用）=====
                 tome_keep_ratio: float = 1.0,
                 tome_sx: int = 2, tome_sy: int = 2,
                 tome_reduce: str = "mean",
                 tome_no_rand: bool = False,
                 tome_seed: int = None,
                 **kwargs):
        super().__init__()

        self.nclass = nclass                                             # Number of classes
        self.input_size = input_size                                     # Number of tokens as input
        self.hidden_dim = hidden_dim                                     # Hidden dimension of the transformer
        self.codebook_size = codebook_size                               # Amount of code in the codebook
        self.proj = proj
        
        self.tome_cfg = ToMeCfg(
            keep_ratio=tome_keep_ratio, sx=tome_sx, sy=tome_sy,
            reduce=tome_reduce, no_rand=tome_no_rand, seed=tome_seed
        )                                                 

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

    def forward(self, x, y, drop_label, mask=None):
        b, h, w = x.size()
        x = x.reshape(b, h*w)

        # label dropout & class embedding
        y = torch.where(drop_label, torch.full_like(y, self.nclass), y)
        y = self.cls_emb(y)

        # pos + tok emb
        pos = torch.arange(0, w*h, dtype=torch.long, device=x.device)
        pos = self.pos_emb(pos)
        x = self.tok_emb(x) + pos

        # 可能的空间下采样（会更新 h, w）
        if self.proj > 1:
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
            x = self.in_proj(x)
            _, _, h, w = x.shape
            x = rearrange(x, 'b c h w -> b (h w) c').contiguous()

        # 注册 token 拼到尾部（不参与合并）
        reg_len = 0
        if self.register > 0:
            reg = torch.arange(0, self.register, dtype=torch.long, device=x.device)
            reg = self.reg_tokens(reg).expand(b, self.register, self.hidden_dim)
            x = torch.cat([x, reg], dim=1)
            reg_len = self.register

        # ToMe 需要的随机数发生器（可设置 seed）
        gen = torch.Generator(device=x.device)
        if self.tome_cfg.seed is not None:
            gen.manual_seed(self.tome_cfg.seed)

        # 进入 Encoder（按层 ToMe：Attn 前后 + MLP 前后）
        x = self.transformer(
            x, y,
            mask=None,                               # 最小改动：mask 在合并下未同步映射，先设 None
            tome_cfg=self.tome_cfg if self.tome_cfg.keep_ratio < 1.0 else None,
            spatial_hw=(h, w),
            reg_len=reg_len,
            tome_generator=gen
        )

        # 丢掉 reg
        x = x[:, :h*w].contiguous()

        # 可能的上采样还原
        if self.proj > 1:
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
            x = self.out_proj(x)
            x = rearrange(x, 'b (c s1 s2) h w -> b (h s1 w s2) c',
                          s1=self.proj, s2=self.proj).contiguous()

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

