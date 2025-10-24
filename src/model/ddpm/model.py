import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .noise_scheduler import NoiseScheduler
from .unet import UNet


class DDPM(nn.Module):
    def __init__(
        self,
        image_shape: tuple[int, int, int],
        num_channels: int = 64,
        ch_mults: list[int] = [1, 2, 2, 4],
        use_attn: list[bool] = [False, False, True, True],
        num_blocks: int = 2,
        num_steps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.image_shape = image_shape
        self.backbone = UNet(
            in_channels=self.image_shape[0],
            num_channels=num_channels,
            ch_mults=ch_mults,
            use_attn=use_attn,
            num_blocks=num_blocks,
        )
        self.noise_scheduler = NoiseScheduler(
            num_steps=num_steps,
            beta_start=beta_start,
            beta_end=beta_end,
        )

    def forward(
        self,
        x: torch.FloatTensor,
        eps: torch.FloatTensor,
        **kwargs,
    ) -> dict[str, torch.FloatTensor]:
        x0 = x

        batch_size = x0.shape[0]
        t = torch.randint(
            low=0,
            high=self.noise_scheduler.num_steps,
            size=(batch_size,),
            device=x0.device,
            dtype=torch.long,
        )
        xt = self.noise(x0=x0, t=t, eps=eps)
        eps_theta = self.backbone(xt, t)
        return {'xt': xt, 'eps_theta': eps_theta}

    def noise(
        self,
        x0: torch.FloatTensor,
        eps: torch.FloatTensor,
        t: torch.LongTensor | None = None,
    ) -> torch.FloatTensor:
        if t is None:
            batch_size = x0.shape[0]
            t = torch.randint(
                low=0,
                high=self.noise_scheduler.num_steps,
                size=(batch_size,),
                device=x0.device,
                dtype=torch.long,
            )
        return self.noise_scheduler(x0=x0, t=t, eps=eps)

    def denoise(self, xt: torch.FloatTensor) -> torch.FloatTensor:
        return self._denoise(xt, 'Denoising...')

    @torch.inference_mode()
    def sample(self, num_samples: int, device: str) -> torch.FloatTensor:
        was_training = self.training
        self.eval()
        xt = torch.randn(num_samples, *self.image_shape, device=device)
        x0 = self._denoise(xt, 'Sampling...')
        self.train(was_training)
        return x0

    @torch.inference_mode()
    def _denoise(self, xt: torch.FloatTensor, desc: str | None = None) -> torch.FloatTensor:
        self.eval()
        batch_size = xt.shape[0]
        T = self.noise_scheduler.num_steps
        for step in tqdm(reversed(range(T)), desc=desc, total=self.noise_scheduler.num_steps):
            t = torch.full((batch_size,), step, device=xt.device, dtype=torch.long)
            eps_theta = self.backbone(xt, t)
            xt = self.noise_scheduler.inverse(
                xt=xt,
                t=t,
                eps=eps_theta,
                add_noise=(step > 0),
            )
        return xt.clamp_(-1.0, 1.0)
