from collections.abc import Mapping as MappingABC
import multiprocessing
from typing import Any, Dict, Optional, Sequence, Tuple
import warnings

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.dataloader import default_collate

def _safe_tensor_stack_collate(batch: Sequence[Any]) -> Any:
    """
    Recursively collate nested samples while stacking tensors directly.

    This avoids DataLoader shared-storage resize paths that can fail for
    tensors backed by non-resizable storages.
    """
    elem = batch[0]

    if torch.is_tensor(elem):
        return torch.stack(list(batch), dim=0)

    if isinstance(elem, MappingABC):
        return {key: _safe_tensor_stack_collate([sample[key] for sample in batch]) for key in elem}

    if isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return type(elem)(*[_safe_tensor_stack_collate(list(samples)) for samples in zip(*batch)])

    if isinstance(elem, tuple):
        return tuple(_safe_tensor_stack_collate(list(samples)) for samples in zip(*batch))

    if isinstance(elem, list):
        return [_safe_tensor_stack_collate(list(samples)) for samples in zip(*batch)]

    return default_collate(list(batch))


def _default_collate_with_fallback(batch: Sequence[Any]) -> Any:
    """
    Use default_collate and fall back to a direct-stack implementation for
    non-resizable storage errors seen with some dataset backends.
    """
    try:
        return default_collate(batch)
    except RuntimeError as exc:
        if "Trying to resize storage that is not resizable" not in str(exc):
            raise
        return _safe_tensor_stack_collate(batch)


class IdentityAllocator:
    """
    A callable class to be used as a collate_fn.
    """
    def __call__(self, batch: Sequence[Any]):
        """
        Collate a batch using the default PyTorch collate.

        :param batch: Sequence of dataset samples.
        :return: Collated batch.
        """
        return _default_collate_with_fallback(batch)


class DataModule(LightningDataModule):
    def __init__(
        self,
        dataset_train: Any = None,
        dataset_val: Any = None,
        dataset_test: Any = None,
        batch_size: int = 16,
        num_workers: int = 16,
        num_val_workers: Optional[int] = None,
        use_costum_ddp_sampler: bool = False,
        prefetch_factor: Optional[int] = None,
        persistent_workers: bool = False,
        shuffle: bool = False
    ):
        """
        Initialize the data module and its datasets/collators.

        :param dataset_train: Training dataset instance.
        :param dataset_val: Validation dataset instance.
        :param dataset_test: Test dataset instance.
        :param batch_size: Batch size for all dataloaders.
        :param num_workers: Number of worker processes for dataloaders.
        :param num_val_workers: Number of workers for validation dataloader.
        :param use_costum_ddp_sampler: Whether to use a custom DDP sampler.
        :param prefetch_factor: Optional prefetch factor for dataloaders.
        :param persistent_workers: Whether to keep dataloader workers alive.
        :param shuffle: Whether to shuffle training dataset.
        :return: None.
        """
        super().__init__()

        self.dataset_train: Any = dataset_train
        self.train_collator: IdentityAllocator = IdentityAllocator()

        self.dataset_val: Any = dataset_val
        self.val_collator: IdentityAllocator = IdentityAllocator()

        self.dataset_test: Any = dataset_test
        self.test_collator: IdentityAllocator = IdentityAllocator()

        self.batch_size: int = batch_size
        self.num_workers: int = num_workers
        self.use_costum_ddp_sampler: bool = use_costum_ddp_sampler
        self.num_val_workers: int = num_workers if num_val_workers is None else num_val_workers
        self.prefetch_factor: Optional[int] = prefetch_factor
        self.persistent_workers: bool = persistent_workers
        self.shuffle: bool = shuffle

        datasets_and_workers = (
            (dataset_train, self.num_workers),
            (dataset_val, self.num_val_workers),
            (dataset_test, self.num_workers),
        )
        rank_sharded_datasets = [
            dataset
            for dataset, _ in datasets_and_workers
            if dataset is not None and getattr(dataset, "distributed_shard", False)
        ]
        if rank_sharded_datasets and self.use_costum_ddp_sampler:
            raise ValueError(
                "Rank-sharded in-memory datasets already contain only this DDP "
                "rank's contiguous items. Set `use_costum_ddp_sampler=false` to "
                "avoid applying a second DistributedSampler."
            )

        fork_only_datasets = [
            dataset
            for dataset, worker_count in datasets_and_workers
            if dataset is not None
            and worker_count > 0
            and getattr(dataset, "requires_fork_workers", False)
        ]
        if fork_only_datasets:
            start_method = multiprocessing.get_context().get_start_method()
            if start_method != "fork":
                raise ValueError(
                    "`InMemoryHealPixLoader` with DataLoader workers requires the "
                    f"`fork` multiprocessing start method, but `{start_method}` is active. "
                    "Set worker counts to zero or use `HealPixLoader`."
                )

        legacy_in_memory_datasets = [
            dataset
            for dataset, worker_count in datasets_and_workers
            if dataset is not None
            and worker_count > 0
            and getattr(dataset, "load_into_memory", False)
            and not getattr(dataset, "requires_fork_workers", False)
        ]
        if legacy_in_memory_datasets:
            warnings.warn(
                "`load_into_memory=True` with DataLoader workers may replicate the "
                "cached dataset in spawned worker processes. Use `num_workers=0` and "
                "`num_val_workers=0` unless the multiprocessing strategy is known to "
                "share memory safely.",
                UserWarning,
            )

    def train_dataloader(self):
        """
        Build the training dataloader.

        :return: Training DataLoader instance.
        """
        if self.use_costum_ddp_sampler:
            sampler = DistributedSampler(dataset=self.dataset_train, shuffle=False)
        else:
            sampler = None
        dataloader = DataLoader(self.dataset_train, sampler=sampler, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.train_collator, prefetch_factor=self.prefetch_factor, persistent_workers=self.persistent_workers, shuffle=self.shuffle)

        return dataloader
    
    def val_dataloader(self):
        """
        Build the validation dataloader.

        :return: Validation DataLoader instance.
        """
        if self.use_costum_ddp_sampler:
            sampler = DistributedSampler(dataset=self.dataset_val, shuffle=False)
        else:
            sampler = None

        dataloader = DataLoader(self.dataset_val, sampler=sampler, batch_size=self.batch_size, num_workers=self.num_val_workers, collate_fn=self.val_collator, prefetch_factor=self.prefetch_factor, persistent_workers=self.persistent_workers)

        return dataloader

    def test_dataloader(self):
        """
        Build the test dataloader.

        :return: Test DataLoader instance.
        """
        if self.use_costum_ddp_sampler:
            sampler = DistributedSampler(dataset=self.dataset_test, shuffle=False)
        else:
            sampler = None

        dataloader = DataLoader(self.dataset_test, sampler=sampler, batch_size=self.batch_size,
                                num_workers=self.num_workers, collate_fn=self.test_collator)

        return dataloader
