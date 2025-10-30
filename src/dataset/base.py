import logging
import random
from typing import Any, Callable

import safetensors.torch
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """
    Base dataset class for loading preprocessed data.

    Parameters
    ----------
    index : list[dict[str, Any]]
        List of data entries with 'path' and optionally 'label' keys.
    limit : int or None, optional
        Maximum number of samples to use.
    shuffle_index : bool, optional
        Whether to shuffle the index (default: False).
    instance_transforms : dict[str, Callable] or None, optional
        Transforms to apply to each instance.
    use_condition : bool, optional
        Whether to use conditional labels (default: True).
    """

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
        """
        Get a single data sample.

        Parameters
        ----------
        index : int
            Index of the sample.

        Returns
        -------
        dict[str, Any]
            Dictionary with 'x' (data) and optionally 'c' (condition).
        """
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
        """
        Get dataset length.

        Returns
        -------
        int
            Number of samples in dataset.
        """
        return len(self._index)

    def load_object(self, path: str) -> torch.Tensor:
        """
        Load data object from file.

        Parameters
        ----------
        path : str
            Path to the safetensors file.

        Returns
        -------
        torch.Tensor
            Loaded tensor data.
        """
        img = safetensors.torch.load_file(path)['tensor']
        return img

    def preprocess_data(self, instance_data: dict[str, Any]) -> dict[str, Any]:
        """
        Apply instance transforms to data.

        Parameters
        ----------
        instance_data : dict[str, Any]
            Dictionary containing data to transform.

        Returns
        -------
        dict[str, Any]
            Transformed data dictionary.
        """
        if self.instance_transforms is not None:
            for name, transform in self.instance_transforms.items():
                if name in instance_data:
                    instance_data[name] = transform(instance_data[name])
        return instance_data

    @staticmethod
    def _filter_records_from_dataset(
        index: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Filter records from dataset index.

        Parameters
        ----------
        index : list[dict[str, Any]]
            Dataset index to filter.

        Returns
        -------
        list[dict[str, Any]]
            Filtered index.
        """
        # Filter logic
        pass

    @staticmethod
    def _assert_index_is_valid(index: list[dict[str, Any]], use_condition: bool) -> None:
        """
        Validate dataset index structure.

        Parameters
        ----------
        index : list[dict[str, Any]]
            Dataset index to validate.
        use_condition : bool
            Whether conditional labels are required.

        Raises
        ------
        AssertionError
            If index structure is invalid.
        """
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
        """
        Sort dataset index.

        Parameters
        ----------
        index : list[dict[str, Any]]
            Dataset index to sort.

        Returns
        -------
        list[dict[str, Any]]
            Sorted index.
        """
        pass

    @staticmethod
    def _shuffle_and_limit_index(
        index: list[dict[str, Any]],
        limit: int | None,
        shuffle_index: bool,
    ) -> list[dict[str, Any]]:
        """
        Shuffle and limit dataset index.

        Parameters
        ----------
        index : list[dict[str, Any]]
            Dataset index to process.
        limit : int or None
            Maximum number of samples to keep.
        shuffle_index : bool
            Whether to shuffle the index.

        Returns
        -------
        list[dict[str, Any]]
            Processed index.
        """
        if shuffle_index:
            random.seed(42)
            random.shuffle(index)

        if limit is not None:
            index = index[:limit]
        return index
