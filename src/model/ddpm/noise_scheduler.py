import torch
import torch.nn as nn
from einops import rearrange


class NoiseScheduler(nn.Module):
    """
    Noise scheduler for DDPM forward and reverse diffusion processes.

    Parameters
    ----------
    num_steps : int
        Number of diffusion timesteps.
    beta_start : float, optional
        Starting beta value (default: 0.0001).
    beta_end : float, optional
        Ending beta value (default: 0.02).
    """

    def __init__(
        self,
        num_steps: int,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.num_steps = num_steps

        beta = torch.linspace(beta_start, beta_end, num_steps)
        self.register_buffer('beta', beta)

        alpha = 1.0 - beta
        self.register_buffer('alpha', alpha)

        alpha_bar = torch.cumprod(alpha, dim=0)
        self.register_buffer('alpha_bar', alpha_bar)

        sigma = torch.sqrt(beta)
        self.register_buffer('sigma', sigma)

    def forward(
        self,
        x0: torch.FloatTensor,
        t: torch.LongTensor,
        eps: torch.FloatTensor | None = None,
    ) -> torch.FloatTensor:
        """
        Forward diffusion process: add noise to clean images.

        Parameters
        ----------
        x0 : torch.FloatTensor
            Clean images of shape (batch, channels, height, width).
        t : torch.LongTensor
            Timesteps of shape (batch,).
        eps : torch.FloatTensor or None, optional
            Noise tensor. If None, samples from standard normal.

        Returns
        -------
        torch.FloatTensor
            Noised images at timestep t.
        """
        if eps is None:
            eps = torch.randn_like(x0)

        alpha_bar_t = rearrange(self.alpha_bar[t], 'b -> b 1 1 1')
        mean = torch.sqrt(alpha_bar_t) * x0
        var = 1 - alpha_bar_t
        return mean + torch.sqrt(var) * eps

    @torch.inference_mode()
    def inverse(
        self,
        xt: torch.FloatTensor,
        t: torch.LongTensor,
        eps: torch.FloatTensor,
        add_noise: bool = True,
    ) -> torch.FloatTensor:
        """
        Reverse diffusion process: denoise images by one timestep.

        Parameters
        ----------
        xt : torch.FloatTensor
            Noisy images at timestep t.
        t : torch.LongTensor
            Current timesteps of shape (batch,).
        eps : torch.FloatTensor
            Predicted noise from the model.
        add_noise : bool, optional
            Whether to add noise during sampling (default: True).

        Returns
        -------
        torch.FloatTensor
            Denoised images at timestep t-1.
        """
        alpha_t = rearrange(self.alpha[t], 'b -> b 1 1 1')
        alpha_bar_t = rearrange(self.alpha_bar[t], 'b -> b 1 1 1')
        beta_t = rearrange(self.beta[t], 'b -> b 1 1 1')

        t_prev = torch.clamp(t - 1, min=0)
        alpha_bar_prev = rearrange(self.alpha_bar[t_prev], 'b -> b 1 1 1')

        coef = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)
        mean = (xt - coef * eps) / torch.sqrt(alpha_t)
        var = ((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)) * beta_t

        if not add_noise:
            return mean

        is_t0 = rearrange((t == 0), 'b -> b 1 1 1')
        var = torch.where(is_t0, torch.zeros_like(var), var)

        z = torch.randn_like(xt)
        return mean + torch.sqrt(torch.clamp(var, min=0.0)) * z
