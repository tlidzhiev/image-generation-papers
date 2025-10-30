import shutil
from pathlib import Path
from typing import Any, Callable, Literal

import safetensors.torch
import torch
import torchvision
from torchvision.transforms import v2
from tqdm.auto import tqdm

from src.dataset.base import BaseDataset
from src.utils.io import get_root, read_json, write_json


class MNISTDataset(BaseDataset):
    """
    MNIST dataset wrapper with preprocessing to safetensors format.

    Parameters
    ----------
    root : Path or str or None, optional
        Root directory for dataset. If None, uses default path.
    split : {'train', 'test'}, optional
        Dataset split to use (default: 'train').
    limit : int or None, optional
        Maximum number of samples to use.
    shuffle_index : bool, optional
        Whether to shuffle the index (default: False).
    instance_transforms : dict[str, Callable] or None, optional
        Transforms to apply to each instance.
    force_reindex : bool, optional
        Force recreation of index file (default: False).
    use_condition : bool, optional
        Whether to use conditional labels (default: True).
    """

    def __init__(
        self,
        root: Path | str | None = None,
        split: Literal['train', 'test'] = 'train',
        limit: int | None = None,
        shuffle_index: bool = False,
        instance_transforms: dict[str, Callable] | None = None,
        force_reindex: bool = False,
        use_condition: bool = True,
    ):
        if root is None:
            root = get_root() / 'data' / 'mnist' / split
        else:
            root = get_root() / root

        index_path = root / 'index.json'
        if index_path.exists() and not force_reindex:
            index = read_json(str(index_path))
        else:
            index = self._create_index(split, root)

        super().__init__(
            index=index,
            limit=limit,
            shuffle_index=shuffle_index,
            instance_transforms=instance_transforms,
            use_condition=use_condition,
        )

    def _create_index(
        self,
        split: Literal['train', 'test'],
        data_path: Path,
    ) -> list[dict[str, Any]]:
        """
        Create dataset index by downloading and preprocessing MNIST.

        Parameters
        ----------
        split : {'train', 'test'}
            Dataset split to process.
        data_path : Path
            Path to save processed data.

        Returns
        -------
        list[dict[str, Any]]
            List of data entries with 'path' and 'label' keys.
        """
        index: list[dict[str, Any]] = []
        data_path.mkdir(exist_ok=True, parents=True)

        transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Lambda(lambda img: img.repeat(3, 1, 1)),  # to compute FID with torchmetrics
            ]
        )
        is_train = split == 'train'

        mnist_dataset = torchvision.datasets.MNIST(
            str(data_path),
            train=is_train,
            download=True,
            transform=transform,
        )

        print(f'Parsing MNIST Dataset metadata for part {split}...')
        for i in tqdm(range(len(mnist_dataset))):
            img, label = mnist_dataset[i]

            save_dict = {'tensor': img}
            save_path = data_path / f'{i:06}.safetensors'
            safetensors.torch.save_file(save_dict, save_path)

            index.append({'path': str(save_path), 'label': int(label)})

        mnist_raw_dir = data_path / 'MNIST'
        if mnist_raw_dir.exists():
            shutil.rmtree(mnist_raw_dir)

        write_json(index, str(data_path / 'index.json'))

        print(f'Successfully processed {len(index)} MNIST images.')
        return index
