import torch
import torch.nn as nn
from einops import rearrange


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.GroupNorm(min(32, in_channels), in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))
        self.conv_block2 = nn.Sequential(
            nn.GroupNorm(min(32, out_channels), out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.FloatTensor, t: torch.FloatTensor) -> torch.FloatTensor:
        h = self.conv_block1(x)
        time_emb = self.time_proj(t)
        time_emb = rearrange(time_emb, 'b d -> b d 1 1')
        h = h + time_emb
        h = self.conv_block2(h)
        return h + self.shortcut(x)
