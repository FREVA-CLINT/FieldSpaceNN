import copy
import torch
from einops import rearrange
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.dataloader import default_collate


class BatchReshapeAllocator:
    """
    A callable class to be used as a collate_fn.
    It accesses the dataset's flag to decide whether to reshape.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def _merge_time_batch_groups(self, source_groups, target_groups, mask_groups, emb_groups, patch_index_zooms):
        n_samples_time = getattr(self.dataset, "load_n_samples_time", 1)


        def _merge_tensor(t):
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

    def __call__(self, batch):
        # Use the default collate function to create the initial batch.
        # This will stack the tensors from __getitem__ along a new dimension.
        # The shape will be (batch_size, n, C, H, W).
        source_zooms_groups_out, target_zooms_groups_out, mask_zooms_groups, emb_groups, patch_index_zooms = default_collate(batch)

        source_zooms_groups_out, target_zooms_groups_out, mask_zooms_groups, emb_groups, patch_index_zooms = self._merge_time_batch_groups(
            source_zooms_groups_out, target_zooms_groups_out, mask_zooms_groups, emb_groups, patch_index_zooms
        )

        return source_zooms_groups_out, target_zooms_groups_out, mask_zooms_groups, emb_groups, patch_index_zooms


class DataModule(LightningDataModule):
    def __init__(self, dataset_train=None, dataset_val=None, dataset_test=None, batch_size=16, num_workers=16, num_val_workers=None, use_costum_ddp_sampler=False, prefetch_factor=None,persistent_workers=False):
        super().__init__()

        self.dataset_train = dataset_train
        self.train_collator = BatchReshapeAllocator(dataset_train)

        self.dataset_val = dataset_val
        self.val_collator = BatchReshapeAllocator(dataset_val)

        self.dataset_test = dataset_test
        self.test_collator = BatchReshapeAllocator(dataset_test)

        self.batch_size=batch_size
        self.num_workers= num_workers
        self.use_costum_ddp_sampler = use_costum_ddp_sampler
        self.num_val_workers = num_workers if num_val_workers is None else num_val_workers
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers

    def train_dataloader(self):
        
        if self.use_costum_ddp_sampler:
            sampler = DistributedSampler(dataset=self.dataset_train, shuffle=False)
        else:
            sampler = None
        dataloader = DataLoader(self.dataset_train, sampler=sampler, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.train_collator, prefetch_factor=self.prefetch_factor, persistent_workers=self.persistent_workers)

        return dataloader
    
    def val_dataloader(self):
        
        if self.use_costum_ddp_sampler:
            sampler = DistributedSampler(dataset=self.dataset_val, shuffle=False)
        else:
            sampler = None

        dataloader = DataLoader(self.dataset_val, sampler=sampler, batch_size=self.batch_size, num_workers=self.num_val_workers, collate_fn=self.val_collator, prefetch_factor=self.prefetch_factor, persistent_workers=self.persistent_workers)

        return dataloader

    def test_dataloader(self):

        if self.use_costum_ddp_sampler:
            sampler = DistributedSampler(dataset=self.dataset_test, shuffle=False)
        else:
            sampler = None

        dataloader = DataLoader(self.dataset_test, sampler=sampler, batch_size=self.batch_size,
                                num_workers=self.num_workers, collate_fn=self.test_collator)

        return dataloader
