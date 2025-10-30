import torch
from torchmetrics.image.fid import FrechetInceptionDistance

from .base import BaseMetric


class FIDMetric(BaseMetric):
    """
    Frechet Inception Distance metric for image quality evaluation.

    Parameters
    ----------
    feature : int
        Inception feature dimension (typically 64, 192, 768, or 2048).
    device : str
        Device to run computations on ('cuda', 'cpu', or 'auto').
    name : str or None, optional
        Name of the metric. If None, uses class name.
    dtype : torch.dtype, optional
        Data type for computations (default: torch.float32).
    """

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
        """
        Update FID metric with fake and real image batches.

        Parameters
        ----------
        fake : torch.FloatTensor
            Generated/fake images.
        real : torch.FloatTensor
            Real images from dataset.
        **kwargs : dict
            Additional keyword arguments (ignored).
        """
        with torch.no_grad():
            fake = (fake - fake.min()) / (fake.max() - fake.min() + 1e-8)
            real = (real - real.min()) / (real.max() - real.min() + 1e-8)

        self.fid.update(fake, real=False)
        self.fid.update(real, real=True)

    def __call__(self) -> float:
        """
        Compute FID score and reset internal state.

        Returns
        -------
        float
            Computed FID score.
        """
        value = self.fid.compute().item()
        self.fid.reset()
        return value

    def __repr__(self) -> str:
        """
        Return string representation of the metric.

        Returns
        -------
        str
            String representation with feature dimension.
        """
        return f'{type(self).__name__}(feature={self.feature})'
