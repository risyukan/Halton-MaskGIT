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

    def forward(self, x, y, drop_label, mask=None, global_masked_token=None, current_mask=None):
        b, h, w = x.size()
        x = x.reshape(b, h*w)

        # Optimization 1: 使用 masked_fill 替代 where + full_like，减少显存分配
        # 假设 drop_label 是 [b] 或标量。如果是 bool 标量，if 判断更快；如果是 tensor，用 masked_fill
        if isinstance(drop_label, torch.Tensor):
             y = y.masked_fill(drop_label, self.nclass)
        elif drop_label:
             y = torch.full_like(y, self.nclass)
        
        y = self.cls_emb(y)

        # Optimization 2: 缓存 pos 或避免重复 device 转换
        # 建议 pos_emb 内部处理，或者这里直接生成。
        pos = torch.arange(0, w*h, device=x.device, dtype=torch.long)
        x = self.tok_emb(x) + self.pos_emb(pos)

        merge_info = None
        if global_masked_token is not None and current_mask is not None:
            # 修正：必须先 .to(x.device) 确保在 GPU 上，然后再 reshape
            gm_flat = global_masked_token.to(x.device).reshape(b, -1).bool()
            cm_flat = current_mask.to(x.device).reshape(b, -1).bool()
            
            # Optimization 3: Meshgrid 计算移出循环 (Batch 共用一套 grid)
            # 如果 h, w 是固定的，这部分甚至可以放在 __init__ 里
            merge_stride = 2
            Wc = (w + merge_stride - 1) // merge_stride
            
            yy, xx = torch.meshgrid(
                torch.arange(h, device=x.device),
                torch.arange(w, device=x.device),
                indexing="ij"
            )
            # cell_flat 只需要计算一次，不需要放在循环里
            cell_flat = ((yy // merge_stride) * Wc + (xx // merge_stride)).reshape(-1)

            # --- ID 生成阶段 (这部分逻辑较复杂，完全向量化较难，保留循环但只做 ID 计算) ---
            batch_cluster_ids = []
            max_tokens = 0
            
            # 这里的循环通常难以避免，因为 unique 和 counts 的数量在不同样本间可能不同
            # 但我们只生成索引，不移动 heavy data (features)
            # --- 3.3: 由于每个样本的 cluster id 一样，改成只算一次 ---
            gm0 = gm_flat[0]          # [hw]
            cm0 = cm_flat[0]          # [hw]

            masked_mask = gm0 & ~cm0
            center_mask = cm0
            keep_mask = ~gm0

            n_keep = int(keep_mask.sum().item())
            n_center = int(center_mask.sum().item())
            n_masked = int(masked_mask.sum().item())

            cluster_ids_1d = torch.empty(h * w, device=x.device, dtype=torch.long)

            # 1) Keep tokens
            if n_keep > 0:
                cluster_ids_1d[keep_mask] = torch.arange(n_keep, device=x.device, dtype=torch.long)

            # 2) Center tokens
            offset = n_keep
            if n_center > 0:
                cluster_ids_1d[center_mask] = torch.arange(n_center, device=x.device, dtype=torch.long) + offset

            # 3) Masked tokens (Grid grouping)
            offset2 = offset + n_center
            if n_masked > 0:
                cell_ids = cell_flat[masked_mask]
                uniq_cell, inv = torch.unique(cell_ids, return_inverse=True)
                cluster_ids_1d[masked_mask] = inv.to(torch.long) + offset2
                L_merge = offset2 + int(uniq_cell.numel())
            else:
                L_merge = offset2

            # 扩展到 batch
            cluster_ids_tensor = cluster_ids_1d.unsqueeze(0).expand(b, -1)


            
            # 记录最大 token 数            
            # 构造全局索引用于 index_add
            # 我们需要把 (batch_idx, token_idx) 映射到平铺的 (batch_idx * L_merge + cluster_id)
            batch_offsets = torch.arange(b, device=x.device) * L_merge
            # global_ids: [b, hw] -> 加上 batch 偏移 -> [b*hw]
            global_ids = (cluster_ids_tensor + batch_offsets.unsqueeze(1)).view(-1)
            
            # 准备输出容器 [b * L_merge, dim]
            flat_x = x.view(-1, x.size(-1))
            merged_x = torch.zeros(b * L_merge, x.size(-1), device=x.device, dtype=x.dtype)
            counts = torch.zeros(b * L_merge, 1, device=x.device, dtype=x.dtype)
            
            # 执行加和聚合 (Vectorized Scatter Add)
            merged_x.index_add_(0, global_ids, flat_x)
            # 计算每个 cluster 有多少个 token 用于平均
            counts.index_add_(0, global_ids, torch.ones_like(flat_x[:, :1]))
            
            # 取平均
            merged_x = merged_x / counts.clamp_min(1.0)
            
            # Reshape 回 [b, L_merge, dim]
            x = merged_x.view(b, L_merge, x.size(-1))

            # 记录用于 Unmerge 的信息
            # 注意：Unmerge 只需要 cluster_ids_tensor 即可，不需要 sizes 列表了
            merge_info = {"cluster_ids": cluster_ids_tensor} 

        base_seq_len = x.shape[1]
        attn_mask = mask

        # Register tokens logic
        if self.register > 0:
            reg = torch.arange(0, self.register, dtype=torch.long, device=x.device)
            reg_emb = self.reg_tokens(reg).expand(b, -1, -1)
            x = torch.cat([x, reg_emb], dim=1)
            if attn_mask is not None:
                # Optimization 5: 使用 F.pad 替代 concat mask，通常更快
                # 假设 mask 是 [b, 1, seq_len, seq_len] 或类似，需根据维度调整
                # 这里保持原逻辑逻辑，但在 mask 最后一维 pad
                attn_mask = F.pad(attn_mask, (0, self.register), value=0)

        x = self.transformer(x, y, mask=attn_mask)

        # drop the register
        x = x[:, :base_seq_len] # 此时通常内存连续，不需要 contiguous() 除非后续显式要求

        # Optimization 6: 向量化 Unmerge (替代 loop)
        if merge_info is not None:
            # x: [b, L_merge, dim]
            # cluster_ids: [b, h*w] -> 对应 x 中的索引
            
            # 我们直接用 gather 就可以把合并后的 token "广播" 回原始位置
            # 需要把 x 扩展对应 cluster_ids 的形状
            ids = merge_info["cluster_ids"].unsqueeze(-1).expand(-1, -1, self.hidden_dim)
            x = torch.gather(x, 1, ids)
            # recovered 已经在 gather 中自动完成，无需预分配 zeros
            
        else:
            x = x[:, :h*w] # .contiguous() 通常可以省略，除非下一层报 stride 错误

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

