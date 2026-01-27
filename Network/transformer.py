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

    def forward(self, x, y, drop_label, mask=None,global_masked_token=None,current_mask=None): #x是code，y是class label，drop_label是bool值
        # global_masked_token形状是 [batch, input_size, input_size] 的布尔张量，表示当前整幅图里哪些 token 还没被预测（全局未解码的位置）
        # current_mask 形状是 [nb_sample, input_size, input_size]，True 代表这一轮要更新的token。
        b, h, w = x.size()
        x = x.reshape(b, h*w)
        # x: [b, h*w]

        # Drop the label if drop_label
        y = torch.where(drop_label, torch.full_like(y, self.nclass), y)
        # torch.full_like,创建一个和 y 形状一样的新张量，里面填满的值是 self.nclass,y是传入的类别标签张量
        # 如果 drop_label 为 True，就把 y 替换成全是 self.nclass 的张量，表示“无类别”。
        # y: [b]
        y = self.cls_emb(y)
        # 把类别标签 y 通过 cls_emb 嵌入成向量表示，形状变成 [b, hidden_dim]

        pos = torch.arange(0, w*h, dtype=torch.long, device=x.device)
        pos = self.pos_emb(pos)

        x = self.tok_emb(x) + pos

        merge_info = None
        if global_masked_token is not None and current_mask is not None:
            gm = global_masked_token.to(x.device).bool()
            cm = current_mask.to(x.device).bool()
            gm_flat = gm.view(b, -1)
            cm_flat = cm.view(b, -1)

            yy, xx = torch.meshgrid(
                torch.arange(h, device=x.device),
                torch.arange(w, device=x.device),
                indexing="ij"
            )
            coords = torch.stack([yy, xx], dim=-1).reshape(-1, 2).float()
            # 例如每 2x2 合并一次（你也可以改成 4 等）
            merge_stride = 2
            Wc = (w + merge_stride - 1) // merge_stride  # ceil(w/stride)

            cell_flat = ((yy // merge_stride) * Wc + (xx // merge_stride)).reshape(-1)  # [h*w]


            merged_batches, clusters, sizes, max_tokens = [], [], [], 0
            for i in range(b):
                masked_idx = torch.nonzero(gm_flat[i] & ~cm_flat[i], as_tuple=False).squeeze(-1)
                center_idx = torch.nonzero(cm_flat[i], as_tuple=False).squeeze(-1)
                keep_idx = torch.nonzero(~gm_flat[i], as_tuple=False).squeeze(-1)


                cluster_ids = torch.empty(h * w, device=x.device, dtype=torch.long)
                # 已解码 token 保持独立
                cluster_ids[keep_idx] = torch.arange(keep_idx.numel(), device=x.device)
                offset = keep_idx.numel()
                # 当前步中心
                cluster_ids[center_idx] = torch.arange(center_idx.numel(), device=x.device) + offset

                # cluster_ids 已经给 keep_idx 和 center_idx 分配好了
                center_num = center_idx.numel()
                offset2 = offset + center_num  # masked clusters 从这里开始编号

                if masked_idx.numel() > 0:
                    # 只对 masked_idx 做网格分组，然后压缩成连续 cluster id
                    cell_ids = cell_flat[masked_idx]  # [n_masked]
                    uniq_cell, inv = torch.unique(cell_ids, return_inverse=True)
                    cluster_ids[masked_idx] = inv + offset2


                minlength = offset2 + (uniq_cell.numel() if masked_idx.numel() > 0 else 0)
                counts = torch.bincount(cluster_ids, minlength=minlength).clamp_min(1)

                merged_batches.append(_merge_tokens_spatial(x[i:i+1], cluster_ids, counts))
                clusters.append(cluster_ids)
                sizes.append(counts.numel())
                max_tokens = max(max_tokens, counts.numel())

            # === no padding version (Halton: same merged length across batch) ===
            L_merge = merged_batches[0].shape[1]
            # 可选：调试断言，确保确实一致
            for i, merged in enumerate(merged_batches):
                assert merged.shape[1] == L_merge, f"merged length mismatch at {i}: {merged.shape[1]} vs {L_merge}"

            # merged_batches 里每个是 [1, L_merge, hidden_dim]，直接 cat 成 [b, L_merge, hidden_dim]
            x = torch.cat(merged_batches, dim=0)

            # 不再需要 pad_mask
            merge_info = {"clusters": clusters, "sizes": sizes}  # sizes 仍然保留给 unmerge 用


        base_seq_len = x.shape[1]
        # 没有 padding 了：如果你原本 mask 只是 padding mask，那这里直接用 None 即可；
        # 但为了兼容你外部可能传入的 mask，最安全是继续用传进来的 mask
        attn_mask = mask


        if self.register > 0:
            reg = torch.arange(0, self.register, dtype=torch.long, device=x.device)
            x = torch.cat([x, self.reg_tokens(reg).expand(b, self.register, self.hidden_dim)], dim=1)
            if attn_mask is not None:
                reg_mask = torch.zeros(b, self.register, device=x.device, dtype=attn_mask.dtype)
                attn_mask = torch.cat([attn_mask, reg_mask], dim=1)

        x = self.transformer(x, y, mask=attn_mask)

        # drop the register
        x = x[:, :base_seq_len].contiguous()

        if merge_info is not None:
            recovered = x.new_zeros(b, h * w, self.hidden_dim)
            for i, cluster_ids in enumerate(merge_info["clusters"]):
                recovered[i:i+1] = _unmerge_tokens_spatial(x[i:i+1, :merge_info["sizes"][i]], cluster_ids)
            x = recovered
        else:
            x = x[:, :h*w].contiguous()

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

