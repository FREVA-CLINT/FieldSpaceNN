import math
from collections import defaultdict
from collections.abc import Mapping as MappingABC
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, DistributedSampler, Sampler
from torch.utils.data.dataloader import default_collate

from ..data.datasets_regular import RegularDataset


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


class _BucketSamplerEpochSetter:
    """Expose a sampler-shaped ``set_epoch`` hook for Lightning."""

    def __init__(self, batch_sampler: "DistributedTimeBucketBatchSampler") -> None:
        self.batch_sampler = batch_sampler

    def set_epoch(self, epoch: int) -> None:
        self.batch_sampler.set_epoch(epoch)


class DistributedTimeBucketBatchSampler(Sampler[List[int]]):
    """Build homogeneous time-length batches and partition them across DDP ranks."""

    def __init__(
        self,
        dataset: Any,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
    ) -> None:
        signatures = getattr(dataset, "sample_time_length_signatures", None)
        if signatures is None:
            raise ValueError(
                "Time-length bucketing requires dataset.sample_time_length_signatures."
            )
        if len(signatures) != len(dataset):
            raise ValueError(
                "Time-length signatures must align one-to-one with dataset samples."
            )
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")

        if num_replicas is None:
            num_replicas = (
                torch.distributed.get_world_size()
                if torch.distributed.is_available() and torch.distributed.is_initialized()
                else 1
            )
        if rank is None:
            rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_available() and torch.distributed.is_initialized()
                else 0
            )
        if num_replicas <= 0:
            raise ValueError(f"num_replicas must be positive, got {num_replicas}.")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank {rank} must be in [0, {num_replicas}).")

        buckets: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
        for index, signature in enumerate(signatures):
            buckets[tuple(int(value) for value in signature)].append(index)

        self.buckets: Dict[Tuple[int, ...], List[int]] = dict(buckets)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.epoch = 0
        # Lightning advances custom batch samplers through
        # ``dataloader.batch_sampler.sampler.set_epoch``.
        self.sampler = _BucketSamplerEpochSetter(self)

    @property
    def global_batch_size(self) -> int:
        return self.batch_size * self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        if self.drop_last:
            return sum(
                len(indices) // self.global_batch_size
                for indices in self.buckets.values()
            )
        return sum(
            math.ceil(len(indices) / self.global_batch_size)
            for indices in self.buckets.values()
        )

    def __iter__(self) -> Iterator[List[int]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        local_batches: List[List[int]] = []

        for signature in sorted(self.buckets):
            bucket_indices = list(self.buckets[signature])
            if self.shuffle and len(bucket_indices) > 1:
                order = torch.randperm(len(bucket_indices), generator=generator).tolist()
                bucket_indices = [bucket_indices[position] for position in order]

            remainder = len(bucket_indices) % self.global_batch_size
            if remainder:
                if self.drop_last:
                    bucket_indices = bucket_indices[:-remainder]
                else:
                    padding_size = self.global_batch_size - remainder
                    repeats = math.ceil(padding_size / len(bucket_indices))
                    bucket_indices.extend((bucket_indices * repeats)[:padding_size])

            rank_start = self.rank * self.batch_size
            rank_end = rank_start + self.batch_size
            for start in range(0, len(bucket_indices), self.global_batch_size):
                global_batch = bucket_indices[start:start + self.global_batch_size]
                local_batches.append(global_batch[rank_start:rank_end])

        if self.shuffle and len(local_batches) > 1:
            order = torch.randperm(len(local_batches), generator=generator).tolist()
            local_batches = [local_batches[position] for position in order]

        yield from local_batches


class BatchReshapeAllocator:
    """
    A callable class to be used as a collate_fn.
    It accesses the dataset's flag to decide whether to reshape.
    """

    def __init__(self, dataset: Any) -> None:
        """
        Initialize the collator with the backing dataset.

        :param dataset: Dataset instance that provides ``load_n_samples_time``.
        :return: None.
        """
        self.dataset: Any = dataset

    def _merge_time_batch_groups(
        self,
        source_groups: Any,
        target_groups: Any,
        mask_groups: Any,
        emb_groups: Any,
        patch_index_zooms: Any
    ):
        """
        Merge the time-sample dimension into the batch dimension when present.

        :param source_groups: Batched source group tensors or nested containers.
        :param target_groups: Batched target group tensors or nested containers.
        :param mask_groups: Batched mask group tensors or nested containers.
        :param emb_groups: Batched embedding group tensors or nested containers.
        :param patch_index_zooms: Patch index mapping or tensor.
        :return: Tuple of merged ``(source_groups, target_groups, mask_groups, emb_groups)``.
            If a tensor has shape ``(b, s, ...)`` with ``s=load_n_samples_time``, it is
            reshaped to ``(b * s, ...)`` so the leading dimension matches the base
            ``(b, v, t, n, d, f)`` convention downstream.
        """
        n_samples_time = getattr(self.dataset, "load_n_samples_time", 1)


        def _merge_tensor(t: torch.Tensor) -> torch.Tensor:
            if t.ndim >= 2 and t.shape[1] == n_samples_time:
                b = t.shape[0]
                return t.reshape(b * n_samples_time, *t.shape[2:])
            return t

        def _merge_obj(obj):
            if torch.is_tensor(obj):
                return _merge_tensor(obj)
            if isinstance(obj, dict):
                return {k: _merge_obj(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_merge_obj(v) for v in obj]
            if isinstance(obj, tuple):
                return tuple(_merge_obj(v) for v in obj)
            return obj

        source_groups = [_merge_obj(group) for group in source_groups]
        target_groups = [_merge_obj(group) for group in target_groups]
        mask_groups = [_merge_obj(group) for group in mask_groups]
        emb_groups = [_merge_obj(group) for group in emb_groups]
        patch_index_zooms = _merge_obj(patch_index_zooms)

        return source_groups, target_groups, mask_groups, emb_groups, patch_index_zooms

    def __call__(self, batch: Sequence[Any]):
        """
        Collate a batch and optionally fold time samples into the batch dimension.

        :param batch: List of dataset samples to collate.
        :return: Collated batch tuple including patch indices. Tensors follow the base
            shape ``(b, v, t, n, d, f)`` after merging the time-sample dimension when
            ``load_n_samples_time > 1``.
        """
        # Use the default collate function to create the initial batch.
        # This will stack the tensors from __getitem__ along a new dimension.
        # The shape will be (batch_size, n, C, H, W).
        source_zooms_groups_out, target_zooms_groups_out, mask_zooms_groups, emb_groups, patch_index_zooms = _default_collate_with_fallback(batch)

        source_zooms_groups_out, target_zooms_groups_out, mask_zooms_groups, emb_groups, patch_index_zooms = self._merge_time_batch_groups(
            source_zooms_groups_out, target_zooms_groups_out, mask_zooms_groups, emb_groups, patch_index_zooms
        )

        return source_zooms_groups_out, target_zooms_groups_out, mask_zooms_groups, emb_groups, patch_index_zooms


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
        shuffle: bool = False,
        bucket_by_time_length: bool = False,
        bucket_seed: int = 0,
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
        :param bucket_by_time_length: Whether to batch samples with identical
            effective temporal lengths using a DDP-aware batch sampler.
        :param bucket_seed: Base seed for deterministic bucket shuffling.
        :return: None.
        """
        super().__init__()

        self.dataset_train: Any = dataset_train
        self.train_collator: BatchReshapeAllocator = IdentityAllocator() if isinstance(dataset_train, RegularDataset) else BatchReshapeAllocator(dataset_train)

        self.dataset_val: Any = dataset_val
        self.val_collator: BatchReshapeAllocator = IdentityAllocator() if isinstance(dataset_train, RegularDataset) else BatchReshapeAllocator(dataset_val)

        self.dataset_test: Any = dataset_test
        self.test_collator: BatchReshapeAllocator = IdentityAllocator() if isinstance(dataset_train, RegularDataset) else BatchReshapeAllocator(dataset_test)

        self.batch_size: int = batch_size
        self.num_workers: int = num_workers
        self.use_costum_ddp_sampler: bool = use_costum_ddp_sampler
        self.num_val_workers: int = num_workers if num_val_workers is None else num_val_workers
        self.prefetch_factor: Optional[int] = prefetch_factor
        self.persistent_workers: bool = persistent_workers
        self.shuffle: bool = shuffle
        self.bucket_by_time_length: bool = bool(bucket_by_time_length)
        self.bucket_seed: int = int(bucket_seed)

    def _bucketed_dataloader(
        self,
        dataset: Any,
        collator: Any,
        num_workers: int,
        shuffle: bool,
    ) -> DataLoader:
        batch_sampler = DistributedTimeBucketBatchSampler(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            seed=self.bucket_seed,
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collator,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
        )

    def train_dataloader(self):
        """
        Build the training dataloader.

        :return: Training DataLoader instance.
        """
        if self.bucket_by_time_length:
            return self._bucketed_dataloader(
                self.dataset_train,
                self.train_collator,
                self.num_workers,
                self.shuffle,
            )

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
        if self.bucket_by_time_length:
            return self._bucketed_dataloader(
                self.dataset_val,
                self.val_collator,
                self.num_val_workers,
                False,
            )

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
        if self.bucket_by_time_length:
            return self._bucketed_dataloader(
                self.dataset_test,
                self.test_collator,
                self.num_workers,
                False,
            )

        if self.use_costum_ddp_sampler:
            sampler = DistributedSampler(dataset=self.dataset_test, shuffle=False)
        else:
            sampler = None

        dataloader = DataLoader(self.dataset_test, sampler=sampler, batch_size=self.batch_size,
                                num_workers=self.num_workers, collate_fn=self.test_collator)

        return dataloader
