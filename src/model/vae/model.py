from typing import Literal

import torch
import torch.nn as nn
from einops import rearrange

from src.utils.torch import get_activation, get_norm_layer, initialize_weights

from .attention import SelfAttentionBlock
from .residual import ResidualBlock


class Encoder(nn.Module):
    def __init__(
        self,
        image_size: int,
        hidden_dims: list[int],
        latent_dim: int,
        activation: Literal['relu', 'leaky_relu', 'silu', 'gelu'] = 'silu',
        norm_type: Literal['batchnorm', 'groupnorm'] = 'groupnorm',
        use_res: bool = True,
        use_attn: bool = True,
        attn_res: list[int] | None = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        if len(hidden_dims) < 2:
            raise ValueError(
                '`hidden_dims` must contain at least two entries: base channels and '
                'at least one downsampling stage.'
            )

        if use_attn and attn_res is None:
            attn_res = [16, 8]

        downsample_dims = hidden_dims[1:]

        self.features = self._build_encoder_layers(
            hidden_dims[0],
            downsample_dims,
            activation,
            norm_type,
            use_res,
            use_attn,
            attn_res,
            image_size,
        )

        self.feature_size = self._calculate_feature_size(image_size, len(downsample_dims))
        self.flatten_dim = downsample_dims[-1] * self.feature_size * self.feature_size

        self.to_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.to_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def _build_encoder_layers(
        self,
        in_channels: int,
        hidden_dims: list[int],
        activation: Literal['relu', 'leaky_relu', 'silu', 'gelu'],
        norm_type: Literal['batchnorm', 'groupnorm'],
        use_res: bool,
        use_attn: bool,
        attn_res: list[int] | None,
        image_size: int,
    ) -> nn.Sequential:
        layers = []
        current_resolution = image_size

        for out_channels in hidden_dims:
            if use_res:
                block = ResidualBlock(
                    in_channels,
                    out_channels,
                    stride=2,
                    activation=activation,
                    norm_type=norm_type,
                )
            else:
                block = nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    get_norm_layer(norm_type, out_channels),
                    get_activation(activation),
                )
            layers.append(block)
            current_resolution = current_resolution // 2
            if use_attn and current_resolution in attn_res:
                layers.append(SelfAttentionBlock(out_channels))
            in_channels = out_channels
        return nn.Sequential(*layers)

    @staticmethod
    def _calculate_feature_size(image_size: int, num_layers: int) -> int:
        downsample_factor = 2**num_layers
        if image_size % downsample_factor != 0:
            raise ValueError(
                f'Encoder configuration expects image size divisible by {downsample_factor}, '
                f'got {image_size}. Adjust `hidden_dims` or `image_size`.'
            )
        return image_size // downsample_factor

    @staticmethod
    def reparameterize(mu: torch.FloatTensor, logvar: torch.FloatTensor) -> torch.FloatTensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, x: torch.FloatTensor
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        features = self.features(x)
        features_flat = rearrange(features, 'b c h w -> b (c h w)')

        mu = self.to_mu(features_flat)
        logvar = self.to_logvar(features_flat)

        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class UpsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: Literal['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'],
    ):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode=mode)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.upsample(x)
        x = self.conv(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        image_size: int,
        hidden_dims: list[int],
        latent_dim: int,
        activation: Literal['relu', 'leaky_relu', 'silu', 'gelu'] = 'relu',
        norm_type: Literal['batchnorm', 'groupnorm'] = 'batchnorm',
        use_res: bool = True,
        use_attn: bool = False,
        attn_res: list[int] | None = None,
        upsample_mode: Literal['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'] = 'nearest',
    ):
        super().__init__()

        if len(hidden_dims) < 2:
            raise ValueError(
                '`hidden_dims` must contain at least two entries: bottleneck channels and '
                'at least one upsampling stage.'
            )

        if use_attn and attn_res is None:
            attn_res = [8, 16]

        upsample_dims = hidden_dims[1:]

        self.initial_size = self._calculate_initial_size(image_size, len(upsample_dims))
        self.initial_channels = hidden_dims[0]
        self.initial_dim = self.initial_channels * self.initial_size * self.initial_size

        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, self.initial_dim),
            get_activation(activation),
        )

        self.features = self._build_decoder_layers(
            self.initial_channels,
            upsample_dims,
            activation,
            norm_type,
            use_res,
            use_attn,
            attn_res,
            self.initial_size,
            upsample_mode,
        )

    @staticmethod
    def _calculate_initial_size(image_size: int, num_layers: int) -> int:
        downsample_factor = 2**num_layers
        if image_size % downsample_factor != 0:
            raise ValueError(
                f'Decoder configuration expects image size divisible by {downsample_factor}, '
                f'got {image_size}. Adjust `hidden_dims` or `image_size`.'
            )
        return image_size // downsample_factor

    def _build_decoder_layers(
        self,
        in_channels: int,
        upsample_dims: list[int],
        activation: Literal['relu', 'leaky_relu', 'silu', 'gelu'],
        norm_type: Literal['batchnorm', 'groupnorm'],
        use_res: bool,
        use_attn: bool,
        attn_res: list[int],
        initial_size: int,
        upsample_mode: Literal['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'],
    ) -> nn.Sequential:
        layers = []
        current_resolution = initial_size

        if use_attn and current_resolution in attn_res:
            layers.append(SelfAttentionBlock(in_channels))

        for out_dim in upsample_dims:
            layers.append(UpsampleBlock(in_channels, out_dim, mode=upsample_mode))
            current_resolution *= 2

            if use_res:
                layers.append(
                    ResidualBlock(
                        out_dim,
                        out_dim,
                        stride=1,
                        activation=activation,
                        norm_type=norm_type,
                    )
                )
            else:
                layers.append(
                    nn.Sequential(
                        get_norm_layer(norm_type, out_dim),
                        get_activation(activation),
                    )
                )
            if use_attn and current_resolution in attn_res:
                layers.append(SelfAttentionBlock(out_dim))
            in_channels = out_dim
        return nn.Sequential(*layers)

    def forward(self, z: torch.FloatTensor) -> torch.FloatTensor:
        hidden = self.from_latent(z)
        hidden = rearrange(
            hidden,
            'b (c h w) -> b c h w',
            c=self.initial_channels,
            h=self.initial_size,
            w=self.initial_size,
        )
        output = self.features(hidden)
        return output


class VAE(nn.Module):
    def __init__(
        self,
        image_size: int,
        in_channels: int,
        hidden_dims: tuple[int, ...] | list[int],
        latent_dim: int,
        activation: Literal['relu', 'leaky_relu', 'silu', 'gelu'] = 'relu',
        norm_type: Literal['batchnorm', 'groupnorm'] = 'batchnorm',
        use_res: bool = True,
        use_attn: bool = False,
        attn_res_enc: list[int] | None = None,
        attn_res_dec: list[int] | None = None,
        upsample_mode: Literal['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'] = 'nearest',
        output_range: Literal['sigmoid', 'tanh'] = 'sigmoid',
        init_mode: Literal['normal', 'uniform'] = 'normal',
    ):
        super().__init__()
        if use_attn and attn_res_enc is None:
            attn_res_enc = [16, 8]
        if use_attn and attn_res_dec is None:
            attn_res_dec = [8, 16]

        self.input_proj = nn.Conv2d(
            in_channels,
            hidden_dims[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.encoder = Encoder(
            image_size=image_size,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            activation=activation,
            norm_type=norm_type,
            use_res=use_res,
            use_attn=use_attn,
            attn_res=attn_res_enc,
        )
        self.decoder = Decoder(
            image_size=image_size,
            hidden_dims=hidden_dims[::-1],
            latent_dim=latent_dim,
            activation=activation,
            norm_type=norm_type,
            use_res=use_res,
            use_attn=use_attn,
            attn_res=attn_res_dec,
            upsample_mode=upsample_mode,
        )

        self.output_proj = nn.Sequential(
            nn.Conv2d(
                hidden_dims[0],
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Sigmoid() if output_range == 'sigmoid' else nn.Tanh(),
        )

        self.latent_dim = latent_dim
        self._initialize_weights(activation, init_mode)

    def forward(self, x: torch.FloatTensor, **kwargs) -> dict[str, torch.FloatTensor]:
        z, mu, logvar = self.encode(x)
        x_hat = self.decode(z)
        return {'x_hat': x_hat, 'mu': mu, 'logvar': logvar}

    def encode(
        self, x: torch.FloatTensor
    ) -> tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        x = self.input_proj(x)
        z, mu, logvar = self.encoder(x)
        return z, mu, logvar

    def decode(self, z: torch.FloatTensor) -> torch.FloatTensor:
        output = self.decoder(z)
        output = self.output_proj(output)
        return output

    @torch.inference_mode()
    def sample(self, num_samples: int, device: str = 'cpu') -> torch.FloatTensor:
        was_training = self.training
        self.eval()
        z = torch.randn(num_samples, self.latent_dim, device=device)
        output = self.decode(z)
        self.train(was_training)
        return output

    def _initialize_weights(self, activation: str, mode: str):
        initialize_weights(self, activation=activation, mode=mode)
