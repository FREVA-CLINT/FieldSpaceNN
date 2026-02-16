from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.dataloader import default_collate

from ..data.datasets_regular import RegularDataset


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
        return default_collate(batch)


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
        source_zooms_groups_out, target_zooms_groups_out, mask_zooms_groups, emb_groups, patch_index_zooms = default_collate(batch)

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
