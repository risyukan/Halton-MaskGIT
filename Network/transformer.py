# Transformer Encoder architecture
# some part have been borrowed from:
#   - NanoGPT: https://github.com/karpathy/nanoGPT
#   - DiT: https://github.com/facebookresearch/DiT

import math

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange


def param_count(archi, model):
    print(f"Size of model {archi}: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 10 ** 6:.3f}M")


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


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

    def forward(self, x, mask=None, active_mask=None):
        """
        active_mask: (b, seq_len) bool — when provided, Q is computed only for
        active (newly-released) positions; K/V use all positions.
        Inactive positions receive a zero attention delta so the residual stream
        is not updated via attention.  active_mask must have the same True-count
        in every row (guaranteed by HaltonSampler's uniform step schedule).
        """
        b, h_w, _ = x.shape

        if active_mask is None:
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
            return proj

        else:
            # ── Q-only-active: Q from U_t tokens, K/V from all tokens ─────
            # active_mask: (b, h_w) bool, uniform True-count across rows
            n_active = int(active_mask[0].sum().item())
            x_active = x[active_mask].view(b, n_active, -1)   # (b, n_active, d)

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

            # Scatter back into a full-size zero tensor.
            # Inactive positions stay zero → residual leaves them unchanged by attn.
            out_full = torch.zeros(b, h_w, proj.shape[-1], device=x.device, dtype=x.dtype)
            out_full[active_mask] = proj.reshape(b * n_active, -1)
            return out_full


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

    def forward(self, x, cond, mask=None, active_mask=None):
        """
        active_mask: (b, seq_len) bool —
        Attention: Q-only-active update when active_mask is provided（active位置のみ更新、inactiveは残差で不変）。
        FFN: 全token更新（active_maskの有無に関わらず常にfull）。
        前半stepでactive_maskを渡す運用を想定: Attention=active-only / FFN=全token。
        """
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.mlp(cond).chunk(6, dim=1)
        # Attention: active-only update when active_mask is provided
        x = x + alpha1.unsqueeze(1) * self.attn(
            modulate(self.ln1(x), gamma1, beta1),
            mask=mask,
            active_mask=active_mask
        )
        # FFN: always full-token update
        x = x + alpha2.unsqueeze(1) * self.ff(modulate(self.ln2(x), gamma2, beta2))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(Block(dim, heads, mlp_dim, dropout=dropout))

    def forward(self, x, cond, mask=None, active_mask=None, partial_update_start_layer=3):
        for i, block in enumerate(self.layers):
            x = block(x, cond, mask=mask, active_mask=active_mask if i >= partial_update_start_layer else None) #只在后面几layer使用partial_update
        return x


class Transformer(nn.Module):
    """ DiT-like transformer with adaLayerNorm with zero initializations """
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

    def forward(self, x, y, drop_label, mask=None, active_mask=None):
        """
        active_mask: (b, h, w) bool — newly-active token positions for
        partial-update mode (Q-only-active attention).
        Pass None for the standard full-update forward pass.
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

        if self.register > 0:
            reg = torch.arange(0, self.register, dtype=torch.long, device=x.device)
            x = torch.cat([x, self.reg_tokens(reg).expand(b, self.register, self.hidden_dim)], dim=1)

        x = self.transformer(x, y, mask=mask, active_mask=seq_active_mask)

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
