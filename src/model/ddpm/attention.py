import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SelfAttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        assert channels % num_heads == 0, 'channels must be divisible by num_heads'

        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        self.proj = nn.Linear(channels, channels, bias=True)

    def forward(
        self, x: torch.FloatTensor, t: torch.FloatTensor | None = None
    ) -> torch.FloatTensor:
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        x_norm = rearrange(x_norm, 'b c h w -> b (h w) c')
        qkv = self.qkv(x_norm)
        qkv = rearrange(
            qkv,
            'b n (qkv heads d) -> qkv b heads n d',
            qkv=3,
            heads=self.num_heads,
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b heads n d -> b n (heads d)')
        out = self.proj(out)
        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)
        return x + out
