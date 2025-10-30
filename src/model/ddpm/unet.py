import math
from typing import Literal

import torch
import torch.nn as nn
from einops import rearrange

from src.utils.torch import initialize_weights

from .attention import SelfAttentionBlock
from .residual import ResidualBlock


class TimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding with learned projection.

    Parameters
    ----------
    dim : int
        Embedding dimension (must be even).

    Raises
    ------
    AssertionError
        If dim is not even.
    """

    def __init__(self, dim: int):
        assert dim % 2 == 0, 'dim must be even'
        super().__init__()

        hidden_dim = dim * 4
        self.proj = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.register_buffer(
            'freq_bands',
            torch.exp(-math.log(10000) * torch.arange(0, dim // 2) / (dim // 2)),
        )

    def forward(self, t: torch.LongTensor) -> torch.FloatTensor:
        """
        Create time embeddings from timesteps.

        Parameters
        ----------
        t : torch.LongTensor
            Timesteps of shape (batch,).

        Returns
        -------
        torch.FloatTensor
            Time embeddings of shape (batch, dim*4).
        """
        freqs = rearrange(t, 'b -> b 1').float() * self.freq_bands
        embeddings = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)
        return self.proj(embeddings)


class DownBlock(nn.Module):
    """
    Downsampling block with residual connection and optional attention.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    time_dim : int
        Time embedding dimension.
    has_attn : bool
        Whether to include self-attention.
    """

    def __init__(self, in_channels: int, out_channels: int, time_dim: int, has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_dim)
        if has_attn:
            self.attn = SelfAttentionBlock(out_channels, num_heads=4)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.FloatTensor, t: torch.FloatTensor) -> torch.FloatTensor:
        """Forward pass through residual and attention layers."""
        x = self.res(x, t)
        x = self.attn(x)
        return x


class MiddleBlock(nn.Module):
    """
    Middle block with two residual blocks and attention.

    Parameters
    ----------
    channels : int
        Number of channels.
    time_dim : int
        Time embedding dimension.
    """

    def __init__(self, channels: int, time_dim: int):
        super().__init__()
        self.res1 = ResidualBlock(channels, channels, time_dim)
        self.attn = SelfAttentionBlock(channels, num_heads=4)
        self.res2 = ResidualBlock(channels, channels, time_dim)

    def forward(self, x: torch.FloatTensor, t: torch.FloatTensor) -> torch.FloatTensor:
        """Forward pass through res-attn-res layers."""
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x


class UpBlock(nn.Module):
    """
    Upsampling block with skip connections and optional attention.

    Parameters
    ----------
    in_channels : int
        Number of input channels (before concatenation).
    out_channels : int
        Number of output channels.
    time_dim : int
        Time embedding dimension.
    has_attn : bool
        Whether to include self-attention.
    """

    def __init__(self, in_channels: int, out_channels: int, time_dim: int, has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_dim)
        if has_attn:
            self.attn = SelfAttentionBlock(out_channels, num_heads=4)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.FloatTensor, t: torch.FloatTensor) -> torch.FloatTensor:
        """Forward pass through residual and attention layers."""
        x = self.res(x, t)
        x = self.attn(x)
        return x


class Downsample(nn.Module):
    """
    Spatial downsampling by factor of 2 using strided convolution.

    Parameters
    ----------
    channels : int
        Number of channels.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(
        self, x: torch.FloatTensor, t: torch.FloatTensor | None = None
    ) -> torch.FloatTensor:
        """Downsample spatial dimensions."""
        return self.conv(x)


class Upsample(nn.Module):
    """
    Spatial upsampling by factor of 2 using nearest neighbor interpolation.

    Parameters
    ----------
    channels : int
        Number of channels.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

    def forward(self, x: torch.FloatTensor, t: torch.FloatTensor) -> torch.FloatTensor:
        """Upsample spatial dimensions."""
        x = self.upsample(x)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    """
    U-Net architecture for DDPM noise prediction.

    Parameters
    ----------
    in_channels : int
        Number of input image channels.
    num_channels : int
        Base number of channels.
    ch_mults : list[int]
        Channel multipliers for each resolution level.
    use_attn : list[bool]
        Whether to use attention at each resolution level.
    num_blocks : int
        Number of residual blocks per resolution level.
    init_mode : {'normal', 'uniform'}, optional
        Weight initialization mode (default: 'normal').
    """

    def __init__(
        self,
        in_channels: int,
        num_channels: int,
        ch_mults: list[int],
        use_attn: list[bool],
        num_blocks: int,
        init_mode: Literal['normal', 'uniform'] = 'normal',
    ):
        super().__init__()
        image_channels = in_channels
        num_resolutions = len(ch_mults)
        self.input_proj = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1)
        self.time_emb = TimeEmbedding(num_channels)

        down = []
        out_channels = in_channels = num_channels
        for i in range(num_resolutions):
            out_channels = in_channels * ch_mults[i]
            for _ in range(num_blocks):
                down.append(DownBlock(in_channels, out_channels, num_channels * 4, use_attn[i]))
                in_channels = out_channels
            if i < num_resolutions - 1:
                down.append(Downsample(in_channels))
        self.down = nn.ModuleList(down)

        self.middle = MiddleBlock(out_channels, num_channels * 4)

        up = []
        in_channels = out_channels
        for i in reversed(range(num_resolutions)):
            out_channels = in_channels
            for _ in range(num_blocks):
                up.append(UpBlock(in_channels, out_channels, num_channels * 4, use_attn[i]))
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, num_channels * 4, use_attn[i]))
            in_channels = out_channels
            if i > 0:
                up.append(Upsample(in_channels))
        self.up = nn.ModuleList(up)

        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, num_channels),
            nn.SiLU(),
            nn.Conv2d(num_channels, image_channels, kernel_size=3, padding=1),
        )
        self._initialize_weights(mode=init_mode)

    def forward(self, x: torch.FloatTensor, t: torch.LongTensor) -> torch.FloatTensor:
        """
        Forward pass through U-Net.

        Parameters
        ----------
        x : torch.FloatTensor
            Input images of shape (batch, channels, height, width).
        t : torch.LongTensor
            Timesteps of shape (batch,).

        Returns
        -------
        torch.FloatTensor
            Predicted noise of shape (batch, channels, height, width).
        """
        x, t = self.input_proj(x), self.time_emb(t)
        skips = [x]
        for block in self.down:
            x = block(x, t)
            skips.append(x)

        x = self.middle(x, t)

        for block in self.up:
            if isinstance(block, Upsample):
                x = block(x, t)
            else:
                s = skips.pop()
                x = torch.cat((x, s), dim=1)
                x = block(x, t)
        return self.output_proj(x)

    def _initialize_weights(self, mode: str):
        """
        Initialize network weights.

        Parameters
        ----------
        mode : str
            Initialization mode ('normal' or 'uniform').
        """
        initialize_weights(self, activation='silu', mode=mode)
