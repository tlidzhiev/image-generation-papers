import torch
import torch.nn as nn
import torch.nn.functional as F


class DDPMLoss(nn.Module):
    """
    Loss function for Denoising Diffusion Probabilistic Models (DDPM).
    Computes MSE between true noise and predicted noise.

    Attributes
    ----------
    loss_names : list[str]
        Names of loss components returned.
    """

    loss_names = ['loss']

    def forward(
        self,
        eps: torch.FloatTensor,
        eps_theta: torch.FloatTensor,
        **kwargs,
    ) -> dict[str, torch.FloatTensor]:
        """
        Compute DDPM loss.

        Parameters
        ----------
        eps : torch.FloatTensor
            True noise added to the input.
        eps_theta : torch.FloatTensor
            Predicted noise from the model.
        **kwargs : dict
            Additional keyword arguments (ignored).

        Returns
        -------
        dict[str, torch.FloatTensor]
            Dictionary with 'loss' key containing the computed loss.
        """
        recon = F.mse_loss(eps, eps_theta, reduction='none').flatten(1)
        recon = recon.sum(dim=1).mean()
        return {'loss': recon}
