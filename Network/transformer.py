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


def param_count(archi, model):
    print(f"Size of model {archi}: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 10 ** 6:.3f}M")


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#以下是tomesd的部分代码
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
        metric = F.normalize(metric, dim=-1, eps=1e-6)
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

    def forward(self, x, cond, mask=None):
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.mlp(cond).chunk(6, dim=1)
        x = x + alpha1.unsqueeze(1) * self.attn(modulate(self.ln1(x), gamma1, beta1), mask=mask)
        x = x + alpha2.unsqueeze(1) * self.ff(modulate(self.ln2(x), gamma2, beta2))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(Block(dim, heads, mlp_dim, dropout=dropout))

    def forward(self, x, cond, mask=None):
        for block in self.layers:
            x = block(x, cond, mask=mask)
        return x


class Transformer(nn.Module):
    """ DiT-like transformer with adaLayerNorm with zero initializations """
    def __init__(self, input_size=16, hidden_dim=768, codebook_size=1024,
                 depth=12, heads=16, mlp_dim=3072, dropout=0., nclass=1000,
                 register=1, proj=1,
                 # ==== ToMeSD 相关参数 ====
                 tome_keep_ratio: float = 1.0,   # 保留多少比例的 image tokens（0~1，1=不开启 ToMe）
                 tome_sx: int = 2,
                 tome_sy: int = 2,
                 tome_no_rand: bool = False,
                 tome_merge_layer_idx: int = 0,   # 在第1个 layer 之前 merge
                 tome_unmerge_before_idx: int = -1,  # 在第几个 layer 之前 unmerge；-1 表示在最后一层前
                 **kwargs):
        super().__init__()

        self.nclass = nclass                       # Number of classes
        self.input_size = input_size               # Number of tokens as input (sqrt)
        self.hidden_dim = hidden_dim               # Hidden dimension of the transformer
        self.codebook_size = codebook_size         # Amount of code in the codebook
        self.proj = proj                           # Projection

        self.cls_emb = nn.Embedding(nclass + 1, hidden_dim)        # Embedding layer for the class token
        self.tok_emb = nn.Embedding(codebook_size + 1, hidden_dim) # Embedding layer for the 'visual' token
        self.pos_emb = nn.Embedding(input_size ** 2, hidden_dim)   # Learnable Positional Embedding

        if self.proj > 1:
            self.in_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=2, stride=2, bias=False)
            self.out_proj = nn.Conv2d(
                hidden_dim, hidden_dim*4, kernel_size=1, stride=1, padding=0, bias=False
            ).to(memory_format=torch.channels_last)

        # The Transformer Encoder a la BERT :)
        self.transformer = TransformerEncoder(
            dim=hidden_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout
        )

        self.last_norm = AdaNorm(x_dim=hidden_dim, y_dim=hidden_dim)   # Last Norm

        self.head = nn.Linear(hidden_dim, codebook_size + 1)
        self.head.weight = self.tok_emb.weight  # weight tied with the tok_emb layer

        self.register = register
        if self.register > 0:
            self.reg_tokens = nn.Embedding(self.register, hidden_dim)

        # ==== ToMeSD 配置 ====
        self.tome_keep_ratio = tome_keep_ratio
        self.tome_sx = tome_sx
        self.tome_sy = tome_sy
        self.tome_no_rand = tome_no_rand

        # layer index 合法化
        self.depth = depth
        self.tome_merge_layer_idx = max(0, min(depth - 1, tome_merge_layer_idx))
        if tome_unmerge_before_idx < 0:
            self.tome_unmerge_before_idx = depth - 1
        else:
            self.tome_unmerge_before_idx = max(0, min(depth - 1, tome_unmerge_before_idx))

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

        # Init register tokens
        if self.register > 0:
            nn.init.normal_(self.reg_tokens.weight, std=0.02)

    def forward(self, x, y, drop_label, mask=None):
        """
        x: [B, H, W]  离散 codebook index
        y: [B]        类别 label
        """
        b, h, w = x.size()
        x = x.reshape(b, h * w)

        # Drop the label if drop_label
        y = torch.where(drop_label, torch.full_like(y, self.nclass), y)
        y = self.cls_emb(y)  # [B, C]

        # position embedding
        pos = torch.arange(0, w * h, dtype=torch.long, device=x.device)
        pos = self.pos_emb(pos)  # [H*W, C]

        x = self.tok_emb(x) + pos  # [B, H*W, C]

        # reshape, proj to smaller space, reshape (patchify!)
        if self.proj > 1:
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
            x = self.in_proj(x)
            _, _, h, w = x.shape                 # 更新下采样后的 h, w
            x = rearrange(x, 'b c h w -> b (h w) c').contiguous()

        img_len = h * w  # 纯 image token 的长度

        # 可选 register tokens（不参与 ToMe merge）
        if self.register > 0:
            reg_idx = torch.arange(0, self.register, dtype=torch.long, device=x.device)
            reg_tokens = self.reg_tokens(reg_idx).expand(b, self.register, self.hidden_dim)
            x = torch.cat([x, reg_tokens], dim=1)  # [B, img_len + R, C]

        # ================== ToMeSD 配置判断 ==================
        # 只在 mask 为空 & keep_ratio 在 (0, 1) 时启用 ToMe
        apply_tome = (
            (mask is None)
            and (self.tome_keep_ratio is not None)
            and (0.0 < self.tome_keep_ratio < 1.0)
        )

        merge_applied = False
        unmerge_done = False
        merge_info = None   # 存 unmerge 函数等信息

        layers = self.transformer.layers
        num_layers = len(layers)

        for i, block in enumerate(layers):

            # ---- 在指定 layer 之前 unmerge ----
            if apply_tome and merge_applied and (not unmerge_done) and (i == self.tome_unmerge_before_idx):
                # 拆成 image / register
                merged_img_len = merge_info["merged_img_len"]
                img_part = x[:, :merged_img_len, :]
                reg_part = x[:, merged_img_len:, :] if self.register > 0 else None

                full_img = merge_info["unmerge_fn"](img_part)  # [B, img_len, C]
                if reg_part is not None:
                    x = torch.cat([full_img, reg_part], dim=1)
                else:
                    x = full_img

                unmerge_done = True

            # ---- 在指定 layer 之前 merge ----
            if apply_tome and (not merge_applied) and (i == self.tome_merge_layer_idx):
                # 当前的 image tokens（注意：可能已经过了前面的 blocks）
                img_part = x[:, :img_len, :]                     # [B, N, C]

                # 根据当前 N 动态算 r
                N = img_len
                r = int(round(N * (1.0 - self.tome_keep_ratio)))
                # 控制在合法范围
                r = max(0, min(N - 1, r))

                if r > 0:
                    metric = img_part                            # 直接用当前 hidden 作为 metric
                    gen = torch.Generator(device=x.device)

                    merge_fn, unmerge_fn = bipartite_soft_matching_random2d(
                        metric=metric,
                        w=w,
                        h=h,
                        sx=self.tome_sx,
                        sy=self.tome_sy,
                        r=r,
                        no_rand=self.tome_no_rand,
                        generator=gen,
                    )

                    img_merged = merge_fn(img_part)              # [B, N - r, C]
                    merged_img_len = img_merged.shape[1]

                    reg_part = x[:, img_len:, :] if self.register > 0 else None
                    if reg_part is not None:
                        x = torch.cat([img_merged, reg_part], dim=1)
                    else:
                        x = img_merged

                    merge_applied = True
                    merge_info = {
                        "unmerge_fn": unmerge_fn,
                        "merged_img_len": merged_img_len,
                        "full_img_len": N,
                    }
                else:
                    # r == 0 相当于不 merge
                    apply_tome = False

            # ---- 正常过 block ----
            x = block(x, y, mask=mask)

        # 如果用户把 unmerge_before_idx 设得很大，导致循环结束还没 unmerge，
        # 这里可以按需要选择是否强制 unmerge 一下（目前我们保持「按配置来」，不强制）。
        if apply_tome and merge_applied and (not unmerge_done):
            # 也可以在这里强制 unmerge（可选）
            merged_img_len = merge_info["merged_img_len"]
            img_part = x[:, :merged_img_len, :]
            reg_part = x[:, merged_img_len:, :] if self.register > 0 else None

            full_img = merge_info["unmerge_fn"](img_part)
            if reg_part is not None:
                x = torch.cat([full_img, reg_part], dim=1)
            else:
                x = full_img

            unmerge_done = True

        # drop the register，只保留 image tokens
        x = x[:, :img_len].contiguous()  # [B, img_len, C]

        # 反投影回原分辨率
        if self.proj > 1:
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
            x = self.out_proj(x)
            x = rearrange(
                x,
                'b (c s1 s2) h w -> b (h s1 w s2) c',
                s1=self.proj,
                s2=self.proj
            ).contiguous()

        x = self.last_norm(x, y)       # [B, img_len(或放大后), C]
        logit = self.head(x)           # [B, img_len(...), codebook_size+1]

        return logit



if __name__ == "__main__":
    from thop import profile

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


