import torch
import torch.nn as nn

from src.utils.torch import get_activation, get_norm_layer


class ResidualBlock(nn.Module):
    """
    Residual block with normalization and activation.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int, optional
        Stride for convolution (default: 1).
    activation : str, optional
        Activation function type (default: 'silu').
    norm_type : str, optional
        Normalization type (default: 'groupnorm').
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        activation: str = 'silu',
        norm_type: str = 'groupnorm',
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
        """
        Forward pass with residual connection.

        Parameters
        ----------
        x : torch.FloatTensor
            Input tensor of shape (batch, in_channels, height, width).

        Returns
        -------
        torch.FloatTensor
            Output tensor of shape (batch, out_channels, height//stride, width//stride).
        """
        out = self.main(x)
        skip = self.skip(x)
        out = out + skip
        return out
