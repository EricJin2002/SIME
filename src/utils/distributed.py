# https://github.com/Lightning-AI/pytorch-lightning/blob/master/src/lightning/fabric/utilities/distributed.py

from collections.abc import Iterable, Iterator, Sized
from typing import TYPE_CHECKING, Any, Optional, Union

import torch
import torch.nn.functional as F
# from lightning_utilities.core.imports import package_available
from torch import Tensor
from torch.utils.data import Dataset, DistributedSampler, Sampler
from typing_extensions import Self, TypeGuard, override

class _DatasetSamplerWrapper(Dataset):
    """Dataset to create indexes from `Sampler` or `Iterable`"""

    def __init__(self, sampler: Union[Sampler, Iterable]) -> None:
        if not isinstance(sampler, Sized):
            raise TypeError(
                "You seem to have configured a sampler in your DataLoader which"
                " does not provide `__len__` method. The sampler was about to be"
                " replaced by `DistributedSamplerWrapper` since `use_distributed_sampler`"
                " is True and you are using distributed training. Either provide `__len__`"
                " method in your sampler, remove it from DataLoader or set `use_distributed_sampler=False`"
                " if you want to handle distributed sampling yourself."
            )
        if len(sampler) == float("inf"):
            raise TypeError(
                "You seem to have configured a sampler in your DataLoader which"
                " does not provide finite `__len__` method. The sampler was about to be"
                " replaced by `DistributedSamplerWrapper` since `use_distributed_sampler`"
                " is True and you are using distributed training. Either provide `__len__`"
                " method in your sampler which returns a finite number, remove it from DataLoader"
                " or set `use_distributed_sampler=False` if you want to handle distributed sampling yourself."
            )
        self._sampler = sampler
        # defer materializing an iterator until it is necessary
        self._sampler_list: Optional[list[Any]] = None

    @override
    def __getitem__(self, index: int) -> Any:
        if self._sampler_list is None:
            self._sampler_list = list(self._sampler)
        return self._sampler_list[index]

    def __len__(self) -> int:
        return len(self._sampler)

    def reset(self) -> None:
        """Reset the sampler list in order to get new sampling."""
        self._sampler_list = list(self._sampler)

class DistributedSamplerWrapper(DistributedSampler):
    """Wrapper over ``Sampler`` for distributed training.

    Allows you to use any sampler in distributed mode. It will be automatically used by Lightning in distributed mode if
    sampler replacement is enabled.

    Note:
        The purpose of this wrapper is to take care of sharding the sampler indices. It is up to the underlying
        sampler to handle randomness and shuffling. The ``shuffle`` and ``seed`` arguments on this wrapper won't
        have any effect.

    """

    def __init__(self, sampler: Union[Sampler, Iterable], *args: Any, **kwargs: Any) -> None:
        super().__init__(_DatasetSamplerWrapper(sampler), *args, **kwargs)

    @override
    def __iter__(self) -> Iterator:
        self.dataset.reset()
        return (self.dataset[index] for index in super().__iter__())
