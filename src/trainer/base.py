from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
import yaml
from numpy import inf
from omegaconf import DictConfig
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.dataset.utils import inf_loop
from src.logger.wandb import WandbWriter
from src.metrics.base import BaseMetric
from src.metrics.tracker import MetricTracker
from src.utils.io import get_root


class BaseTrainer:
    """
    Base trainer class for training neural networks.

    Parameters
    ----------
    cfg : DictConfig
        Experiment config containing training config.
    device : str
        Device for training ('cuda' or 'cpu').
    dataloaders : dict[str, DataLoader]
        Dictionary of dataloaders by split.
    model : nn.Module
        Model to train.
    criterion : nn.Module
        Loss function.
    metrics : dict[str, list[BaseMetric]]
        Metrics to track.
    optimizer : Optimizer
        Optimizer for training.
    lr_scheduler : LRScheduler
        Learning rate scheduler.
    logger : Logger
        Logger for messages.
    writer : WandbWriter
        W&B writer for logging.
    batch_transforms : dict[str, nn.Module or None]
        Batch transformations.
    skip_oom : bool, optional
        Skip batches on OOM error (default: True).
    epoch_len : int or None, optional
        Number of steps in each epoch for iteration-based training. If None, use epoch-based training (len(dataloader)).
    """

    def __init__(
        self,
        cfg: DictConfig,
        device: str,
        dataloaders: dict[str, DataLoader],
        model: nn.Module,
        criterion: nn.Module,
        metrics: dict[str, list[BaseMetric]],
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        logger: Logger,
        writer: WandbWriter,
        batch_transforms: dict[str, nn.Module | None],
        skip_oom: bool = True,
        epoch_len: int | None = None,
    ):
        self.is_train = True

        self.cfg = cfg
        self.cfg_trainer = self.cfg.trainer

        self.device = device
        self.skip_oom = skip_oom

        self.logger = logger
        self.log_step = self.cfg_trainer.get('log_step', 50)

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.batch_transforms = batch_transforms

        # define dataloaders
        if epoch_len is None:
            self.train_dataloader = dataloaders['train']
            self.epoch_len = len(self.train_dataloader)
        else:
            self.train_dataloader = inf_loop(dataloaders['train'])
            self.epoch_len = epoch_len
        self.eval_dataloaders = {k: v for k, v in dataloaders.items() if k != 'train'}

        # define epochs
        self._last_epoch = 0  # required for saving on interruption
        self.start_epoch = 1
        self.num_epochs = self.cfg_trainer.num_epochs

        # configuration to monitor model performance and save best
        self.save_period = self.cfg_trainer.save_period  # checkpoint each save_period epochs
        self.monitor = self.cfg_trainer.get('monitor', 'off')  # format: "mnt_mode mnt_metric"

        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = self.cfg_trainer.get('early_stop', inf)
            if self.early_stop is None or self.early_stop <= 0:
                self.early_stop = inf

        # setup visualization writer instance
        self.writer = writer

        # define metrics
        self.metrics = metrics
        self.train_metrics = MetricTracker(
            *self.criterion.loss_names,
            'grad_norm',
            *[m.name for m in self.metrics['train']],
        )
        self.eval_metrics = MetricTracker(
            *self.criterion.loss_names,
            *[m.name for m in self.metrics['inference']],
        )

        # define checkpoint dir and init everything if required
        self.checkpoint_dir: Path = get_root() / self.cfg_trainer.checkpoint_dir
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

    def train(self):
        """
        Main training loop with keyboard interrupt handling.

        Raises
        ------
        KeyboardInterrupt
            Re-raised after saving checkpoint.
        """
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self._log('Saving model on keyboard interrupt')
            self._save_checkpoint(self._last_epoch, save_best=False)
            raise e

    def _train_process(self):
        """Execute the full training process across all epochs."""
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.num_epochs + 1):
            self._last_epoch = epoch
            result = self._train_epoch(epoch)

            # save logged information into logs dict
            logs = {'epoch': epoch}
            logs.update(result)

            # print logged information to the screen
            self._log(f'\nMetrics:\n{yaml.dump(logs)}')

            # evaluate model performance according to configured metric,
            # save best checkpoint as model_best
            best, stop_process, not_improved_count = self._monitor_performance(
                logs, not_improved_count
            )

            if epoch % self.save_period == 0 or best or epoch == self.num_epochs:
                self._save_checkpoint(epoch, save_best=best, only_best=True)

            if stop_process:  # early_stop
                break

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        """
        Training logic for an epoch.

        Parameters
        ----------
        epoch : int
            Current training epoch number.

        Returns
        -------
        dict[str, float]
            Dictionary containing average loss and metrics for this epoch.
        """
        self.is_train = True
        self.model.train()
        self.train_metrics.reset()
        self.writer.set_step((epoch - 1) * self.epoch_len)
        self.writer.add_scalar('epoch', epoch)

        pbar = tqdm(self.train_dataloader, desc=f'Train Epoch {epoch}', total=self.epoch_len)
        for batch_idx, batch in enumerate(pbar):
            try:
                batch = self.process_batch(
                    batch,
                    metrics=self.train_metrics,
                )
            except torch.cuda.OutOfMemoryError as e:
                if self.skip_oom:
                    self.logger.warning('OOM on batch. Skipping batch.')
                    torch.cuda.empty_cache()  # free some memory
                    continue
                else:
                    raise e

            pbar.set_postfix({'loss': batch['loss'].item()})
            self.train_metrics.update('grad_norm', self._get_grad_norm())

            # log current results
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.epoch_len + batch_idx)
                self.writer.add_scalar('lr', self.lr_scheduler.get_last_lr()[0])
                self._log_scalars(self.train_metrics)
                self._log_batch(batch_idx, batch, epoch)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()

            if batch_idx + 1 > self.epoch_len:
                break

        logs = last_train_metrics

        # Run val/test
        for part, dataloader in self.eval_dataloaders.items():
            val_logs = self._evaluation_epoch(epoch, part, dataloader)
            logs.update(**{f'{part}_{name}': value for name, value in val_logs.items()})

        return logs

    @torch.no_grad()
    def _evaluation_epoch(self, epoch: int, part: str, dataloader: DataLoader) -> dict[str, float]:
        """
        Evaluate model on a partition after training for an epoch.

        Parameters
        ----------
        epoch : int
            Current training epoch number.
        part : str
            Partition name to evaluate on ('val', 'test', etc.).
        dataloader : DataLoader
            DataLoader for the partition.

        Returns
        -------
        dict[str, float]
            Dictionary containing evaluation metrics.
        """
        self.is_train = False
        self.model.eval()
        self.eval_metrics.reset()

        for batch_idx, batch in tqdm(
            enumerate(dataloader),
            desc=f'{part.capitalize()} Epoch {epoch}',
            total=len(dataloader),
        ):
            batch = self.process_batch(
                batch,
                metrics=self.eval_metrics,
                part=part,
            )

        self.writer.set_step(epoch * self.epoch_len, part)
        if part == 'test':  # compute FID only on test set
            for metric in self.metrics['inference']:
                self.eval_metrics.update(metric.name, metric())
        self._log_scalars(self.eval_metrics)
        self._log_batch(batch_idx, batch, epoch)
        return self.eval_metrics.result()

    def process_batch(
        self,
        batch: dict[str, Any],
        metrics: MetricTracker,
    ) -> dict[str, Any]:
        """
        Process a single batch (forward, backward, metrics).

        Parameters
        ----------
        batch : dict[str, Any]
            Batch data.
        metrics : MetricTracker
            Metrics tracker to update.

        Returns
        -------
        dict[str, Any]
            Processed batch with outputs and losses.

        Raises
        ------
        NotImplementedError
            Must be implemented in subclass.
        """
        raise NotImplementedError('This method must be implemented in the nested class')

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

        Raises
        ------
        NotImplementedError
            Must be implemented in subclass.
        """
        raise NotImplementedError('This method must be implemented in the nested class')

    def _monitor_performance(
        self,
        logs: dict[str, float],
        not_improved_count: int,
    ) -> tuple[bool, bool, int]:
        """
        Check if there is improvement in metrics for early stopping and best checkpoint saving.

        Parameters
        ----------
        logs : dict[str, float]
            Logs after training and evaluating the model for an epoch.
        not_improved_count : int
            Current number of epochs without improvement.

        Returns
        -------
        best : bool
            True if the monitored metric has improved.
        stop_process : bool
            True to stop the process (early stopping). Metric did not improve for too many epochs.
        not_improved_count : int
            Updated number of epochs without improvement.
        """
        best = False
        stop_process = False
        if self.mnt_mode != 'off':
            try:
                # check whether model performance improved or not,
                # according to specified metric(mnt_metric)
                if self.mnt_mode == 'min':
                    improved = logs[self.mnt_metric] <= self.mnt_best
                elif self.mnt_mode == 'max':
                    improved = logs[self.mnt_metric] >= self.mnt_best
                else:
                    improved = False
            except KeyError:
                self._log(
                    message=f"Warning: Metric '{self.mnt_metric}' is not found. "
                    'Model performance monitoring is disabled.',
                    message_type='WARNING',
                )
                self.mnt_mode = 'off'
                improved = False

            if improved:
                self.mnt_best = logs[self.mnt_metric]
                not_improved_count = 0
                best = True
            else:
                not_improved_count += 1

            if not_improved_count >= self.early_stop:
                self._log(
                    f"Validation performance didn't improve for {self.early_stop} epochs. Training stops."
                )
                stop_process = True
        return best, stop_process, not_improved_count

    def _to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Move necessary tensors in batch to the device.

        Parameters
        ----------
        batch : dict[str, Any]
            Dictionary-based batch containing data from dataloader.

        Returns
        -------
        dict[str, Any]
            Batch with specified tensors moved to device.
        """
        for key in self.cfg_trainer.device_tensors:
            batch[key] = batch[key].to(self.device)
        return batch

    def _transform_batch(
        self,
        batch: dict[str, Any],
        transform_type: Literal['train', 'inference', 'sampling'] | None = None,
    ) -> dict[str, Any]:
        """
        Apply batch transforms on the whole batch.

        Like instance transforms in BaseDataset, but for the whole batch.
        Improves pipeline speed, especially with GPU.

        Parameters
        ----------
        batch : dict[str, Any]
            Dictionary-based batch containing data from dataloader.
        transform_type : {'train', 'inference', 'sampling'} or None, optional
            Type of transform to apply. If None, inferred from self.is_train.

        Returns
        -------
        dict[str, Any]
            Batch with transforms applied.
        """
        if transform_type is None:
            transform_type = 'train' if self.is_train else 'inference'

        transforms = self.batch_transforms.get(transform_type)

        if transforms is None:
            return batch

        for transform_name, transform_fn in transforms.items():
            batch[transform_name] = transform_fn(batch[transform_name])
        return batch

    def _clip_grad_norm(self):
        """
        Clip gradient norm by value in config.trainer.max_grad_norm.
        """
        if self.cfg_trainer.get('max_grad_norm', None) is not None:
            clip_grad_norm_(self.model.parameters(), self.cfg_trainer.max_grad_norm)

    @torch.no_grad()
    def _get_grad_norm(self, norm_type: float | str | None = 2) -> float:
        """
        Calculate gradient norm for logging.

        Parameters
        ----------
        norm_type : float or str or None, optional
            Order of the norm (default: 2).

        Returns
        -------
        float
            Calculated gradient norm.
        """
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        """
        Log all metrics from metric tracker to experiment writer.

        Parameters
        ----------
        metric_tracker : MetricTracker
            Tracker containing calculated metrics.
        """
        for metric_name in metric_tracker.names():
            self.writer.add_scalar(f'{metric_name}', metric_tracker[metric_name])

    def _save_checkpoint(self, epoch: int, save_best: bool = False, only_best: bool = False):
        """
        Save model checkpoint.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        save_best : bool, optional
            If True, rename checkpoint to 'model_best.pth' (default: False).
        only_best : bool, optional
            If True and checkpoint is best, save only as 'model_best.pth'
            without epoch-numbered duplicate (default: False).
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'monitor_best': f'{self.mnt_metric}: {self.mnt_best}',
            'cfg': self.cfg,
        }
        filename = str(self.checkpoint_dir / f'checkpoint-epoch-{epoch}.pth')
        if not (only_best and save_best):
            torch.save(state, filename)
            self._log(f'Saving checkpoint: {filename} ...')
            self.writer.add_checkpoint(filename, str(self.checkpoint_dir.parent))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.writer.add_checkpoint(best_path, str(self.checkpoint_dir.parent))
            self._log('Saving current best: model_best.pth ...')

    def _log(self, message: str, message_type: Literal['INFO', 'WARNING', 'DEBUG'] = 'INFO'):
        """
        Log a message using the configured logger.

        Parameters
        ----------
        message : str
            Message to log.
        message_type : {'INFO', 'WARNING', 'DEBUG'}, optional
            Type of log message (default: 'INFO').
        """
        message = f'{type(self).__name__} {message}'
        if self.logger is not None:
            match message_type:
                case 'INFO':
                    self.logger.info(message)
                case 'DEBUG':
                    self.logger.debug(message)
                case 'WARNING':
                    self.logger.warning(message)
                case _:
                    self.logger.info(message)
        else:
            print(f'{datetime.now()} {message_type}: {message}')
