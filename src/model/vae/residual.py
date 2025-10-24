import torch
import torch.nn as nn

from src.utils.torch import get_activation, get_norm_layer


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        activation: str = "silu",
        norm_type: str = "groupnorm",
    ):
        super().__init__()

        self.main = nn.Sequential(
            get_norm_layer(norm_type, in_channels),
            get_activation(activation),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            get_norm_layer(norm_type, out_channels),
            get_activation(activation),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )

        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        out = self.main(x)
        skip = self.skip(x)
        out = out + skip
        return out
