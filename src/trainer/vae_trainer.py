from typing import Any, Literal

import torch
import torchvision

from src.loss.vae import VAELoss
from src.metrics.tracker import MetricTracker
from src.model.vae import VAE

from .base import BaseTrainer


class VAETrainer(BaseTrainer):
    """
    Trainer for VAE models.
    """

    model: VAE
    criterion: VAELoss

    def process_batch(
        self,
        batch: dict[str, Any],
        metrics: MetricTracker,
        part: Literal['train', 'val', 'test'] = 'train',
    ) -> dict[str, Any]:
        """
        Process batch for VAE training/evaluation.

        Parameters
        ----------
        batch : dict[str, Any]
            Batch with 'x' (images).
        metrics : MetricTracker
            Metrics tracker.
        part : {'train', 'val', 'test'}, optional
            Current phase (default: 'train').

        Returns
        -------
        dict[str, Any]
            Batch with outputs, losses, and optional samples.
        """
        batch = self._to_device(batch)
        batch = self._transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics['train' if part == 'train' else 'inference']
        if part == 'train':
            self.optimizer.zero_grad()

        output = self.model(**batch)
        batch.update(output)
        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if part == 'train':
            batch['loss'].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()
            self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.criterion.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        # compute FID only on test set
        if part == 'test':
            # get real images
            real = batch['x'].clone().detach()
            real = self._transform_batch(batch={'x': real}, transform_type='sampling')['x']

            # generate fake samples
            fake = self.model.sample(num_samples=real.shape[0], device=real.device)
            fake = self._transform_batch(batch={'x': fake}, transform_type='sampling')['x']

            # update metrics (FID accumulates Inception features)
            for metric in metric_funcs:
                metric.update(fake=fake, real=real)

            # store samples for logging
            batch['sample'] = fake

        return batch

    @torch.no_grad()
    def _log_batch(self, batch_idx: int, batch: dict[str, Any], epoch: int):
        """
        Log batch results (images, etc.).

        Parameters
        ----------
        batch_idx : int
            Batch index.
        batch : dict[str, Any]
            Batch data and outputs.
        epoch : int
            Current epoch number.
        """
        b, c, h, w = batch['x'].shape
        num_samples = min(8, b)

        indices = torch.randperm(b)[:num_samples]
        originals = batch['x'][indices].detach()
        reconstructions = batch['x_hat'][indices].detach()

        def make_grid_block(images: torch.FloatTensor, title: str, nrow: int):
            grid = torchvision.utils.make_grid(
                images,
                nrow=nrow,
                normalize=True,
                pad_value=1.0,
            )
            self.writer.add_image(title, grid)

        if batch.get('sample') is None:
            orig_rows = originals.view(4, 2, c, h, w)
            recon_rows = reconstructions.view(4, 2, c, h, w)
            rows = torch.cat([orig_rows, recon_rows], dim=1)
            images = rows.reshape(-1, c, h, w)
            make_grid_block(images, f'Epoch {epoch}: Original — Reconstructed', nrow=4)
        else:
            sampled = batch['sample'][indices]
            orig_rows = originals.view(4, 2, c, h, w)
            recon_rows = reconstructions.view(4, 2, c, h, w)
            sample_rows = sampled.view(4, 2, c, h, w)
            rows = torch.cat([orig_rows, recon_rows, sample_rows], dim=1)
            images = rows.reshape(-1, c, h, w)
            make_grid_block(images, f'Epoch {epoch}: Original — Reconstructed — Generated', nrow=6)
