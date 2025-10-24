import logging
import random
from typing import Any, Callable

import safetensors.torch
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(
        self,
        index: list[dict[str, Any]],
        limit: int | None = None,
        shuffle_index: bool = False,
        instance_transforms: dict[str, Callable] | None = None,
        use_condition: bool = True,
    ) -> None:
        self._assert_index_is_valid(index, use_condition)

        index = self._shuffle_and_limit_index(index, limit, shuffle_index)
        self._index: list[dict[str, Any]] = index

        self.instance_transforms = instance_transforms
        self.use_condition = use_condition

    def __getitem__(self, index: int) -> dict[str, Any]:
        data_dict = self._index[index]
        data_path = data_dict['path']
        data_object = self.load_object(data_path)

        if self.use_condition:
            data_label = data_dict['label']
            instance_data = {'x': data_object, 'c': data_label}
        else:
            instance_data = {'x': data_object}

        instance_data = self.preprocess_data(instance_data)
        return instance_data

    def __len__(self) -> int:
        return len(self._index)

    def load_object(self, path: str) -> torch.Tensor:
        img = safetensors.torch.load_file(path)['tensor']
        return img

    def preprocess_data(self, instance_data: dict[str, Any]) -> dict[str, Any]:
        if self.instance_transforms is not None:
            for name, transform in self.instance_transforms.items():
                if name in instance_data:
                    instance_data[name] = transform(instance_data[name])
        return instance_data

    @staticmethod
    def _filter_records_from_dataset(
        index: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        # Filter logic
        pass

    @staticmethod
    def _assert_index_is_valid(index: list[dict[str, Any]], use_condition: bool) -> None:
        for entry in index:
            assert 'path' in entry, (
                "Each dataset item should include field 'path' - path to image file."
            )
            if use_condition:
                assert 'label' in entry, (
                    "Each dataset item should include field 'label' - "
                    'object ground-truth label (required when use_condition=True).'
                )

    @staticmethod
    def _sort_index(index: list[dict[str, Any]]) -> list[dict[str, Any]]:
        pass

    @staticmethod
    def _shuffle_and_limit_index(
        index: list[dict[str, Any]],
        limit: int | None,
        shuffle_index: bool,
    ) -> list[dict[str, Any]]:
        if shuffle_index:
            random.seed(42)
            random.shuffle(index)

        if limit is not None:
            index = index[:limit]
        return index
