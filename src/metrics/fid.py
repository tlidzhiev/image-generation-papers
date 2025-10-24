import torch
from torchmetrics.image.fid import FrechetInceptionDistance

from .base import BaseMetric


class FIDMetric(BaseMetric):
    def __init__(
        self,
        feature: int,
        device: str,
        name: str | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(name=name)
        self.feature = feature
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.fid = (
            FrechetInceptionDistance(
                feature=feature,
                normalize=True,
            )
            .to(device)
            .set_dtype(dtype)
        )

    def update(self, fake: torch.FloatTensor, real: torch.FloatTensor, **kwargs):
        with torch.no_grad():
            fake = (fake - fake.min()) / (fake.max() - fake.min() + 1e-8)
            real = (real - real.min()) / (real.max() - real.min() + 1e-8)

        self.fid.update(fake, real=False)
        self.fid.update(real, real=True)

    def __call__(self) -> float:
        value = self.fid.compute().item()
        self.fid.reset()
        return value

    def __repr__(self) -> str:
        return f'{type(self).__name__}(feature={self.feature})'
