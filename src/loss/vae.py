from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAELoss(nn.Module):
    def __init__(
        self,
        beta: float = 1.0,
        reconstruction_loss: Literal['mse', 'bce'] = 'mse',
        per_element_mean: bool = False,  # False: sum over dims, then mean over batch
    ) -> None:
        super().__init__()
        self.beta = beta
        self.reconstruction_loss = reconstruction_loss
        self.per_element_mean = per_element_mean
        self.loss_names = ['loss', 'recon_loss', 'kl_loss']

    def forward(
        self,
        x: torch.FloatTensor,
        x_hat: torch.FloatTensor,
        mu: torch.FloatTensor,
        logvar: torch.FloatTensor,
    ) -> dict[str, torch.FloatTensor]:
        if self.reconstruction_loss == 'mse':
            per_sample = F.mse_loss(x_hat, x, reduction='none').flatten(1)
            recon = per_sample.mean(1) if self.per_element_mean else per_sample.sum(1)
        elif self.reconstruction_loss == 'bce':
            per_sample = F.binary_cross_entropy(x_hat, x, reduction='none').flatten(1)
            recon = per_sample.mean(1) if self.per_element_mean else per_sample.sum(1)
        else:
            raise ValueError(
                f"reconstruction_loss must be 'mse' or 'bce', got {self.reconstruction_loss}"
            )
        recon = recon.mean()  # mean over batch

        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl = kl.flatten(1).sum(1).mean()  # sum over latent dim, mean over batch

        loss = recon + self.beta * kl
        return {'loss': loss, 'recon_loss': recon, 'kl_loss': kl}

    def __repr__(self):
        return (
            f'{type(self).__name__}(beta={self.beta}, '
            f'reconstruction_loss="{self.reconstruction_loss}", '
            f'per_element_mean={self.per_element_mean})'
        )
