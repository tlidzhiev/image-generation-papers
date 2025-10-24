import torch
import torch.nn as nn
import torch.nn.functional as F


class DDPMLoss(nn.Module):
    loss_names = ['loss']

    def forward(
        self,
        eps: torch.FloatTensor,
        eps_theta: torch.FloatTensor,
        **kwargs,
    ) -> dict[str, torch.FloatTensor]:
        recon = F.mse_loss(eps, eps_theta, reduction='none').flatten(1)
        recon = recon.sum(dim=1).mean()
        return {'loss': recon}
