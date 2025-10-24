from itertools import repeat

import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader


def inf_loop(dataloader: DataLoader):
    for loader in repeat(dataloader):
        yield from loader


def _create_dataloader(cfg: DictConfig, split: str) -> DataLoader:
    dataset = instantiate(cfg.dataset[split])
    dataloader = instantiate(cfg.dataloader[split], dataset=dataset)
    return dataloader


def _to_device(
    batch_transforms: dict[str, dict[str, nn.Sequential]],
    device: str,
):
    for split, transforms in batch_transforms.items():
        if transforms is not None:
            for tensor_name, transform in transforms.items():
                transforms[tensor_name] = transform.to(device)


def get_dataloaders(
    cfg: DictConfig, device: str
) -> tuple[
    dict[str, DataLoader],
    dict[str, dict[str, nn.Sequential]],
]:
    batch_transforms = instantiate(cfg.transforms.batch_transforms, device)
    _to_device(batch_transforms, device)

    dataloaders = {
        'train': _create_dataloader(cfg, 'train'),
    }
    if cfg.dataset.get('val') is not None:
        dataloaders['val'] = _create_dataloader(cfg, 'val')
    if cfg.dataset.get('test') is not None:
        dataloaders['test'] = _create_dataloader(cfg, 'test')

    return dataloaders, batch_transforms
