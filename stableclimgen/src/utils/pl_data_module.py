from lightning.pytorch import Trainer, LightningDataModule
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.dataloader import default_collate
import torch

class CoarsenLevelCollator:
    """
    A callable class to be used as a collate_fn.
    It accesses the dataset's flag to decide whether to reshape.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self, batch):
        # Use the default collate function to create the initial batch.
        # This will stack the tensors from __getitem__ along a new dimension.
        # The shape will be (batch_size, n, C, H, W).
        collated_batch = default_collate(batch)

        if self.dataset.coarsen_lvl_single_map != -1:
            data_source, data_target, coords_input, coords_output, indices_sample, drop_mask, embed_data, dists_input = collated_batch
            b, nt, n, nh, nv, nc = data_source.shape

            data_source = data_source.view(-1, nt, 4**self.dataset.coarsen_lvl_single_map, nh, nv, nc)
            data_target = data_target.view(-1, nt, 4**self.dataset.coarsen_lvl_single_map, nh, nv, nc)
            drop_mask = drop_mask.view(-1, nt, 4**self.dataset.coarsen_lvl_single_map, nh, nv)

            nr = data_source.shape[0] // b

            if coords_input.numel() != 0:
                coords_input = coords_input.view(-1, nt, 4**self.dataset.coarsen_lvl_single_map)
            if coords_output.numel() != 0:
                coords_output = coords_output.view(-1, nt, 4**self.dataset.coarsen_lvl_single_map)
            if dists_input.numel() != 0:
                dists_input = dists_input.view(-1, 4**self.dataset.coarsen_lvl_single_map)
            indices_sample = {'sample': torch.arange(nr).unsqueeze(-1).repeat(b, nt),
                              'sample_level': torch.tensor(self.dataset.coarsen_lvl_single_map).unsqueeze(-1).repeat(data_source.shape[0], nt)}
            embed_data['VariableEmbedder'] = embed_data['VariableEmbedder'].repeat_interleave(nr, dim=0)
            embed_data['TimeEmbedder'] = embed_data['TimeEmbedder'].repeat_interleave(nr, dim=0)

            collated_batch = data_source, data_target, coords_input, coords_output, indices_sample, drop_mask, embed_data, dists_input

        return collated_batch


class DataModule(LightningDataModule):
    def __init__(self, dataset_train=None, dataset_val=None, dataset_test=None, batch_size=16, num_workers=16, use_costum_ddp_sampler=False, collate_fn=None):
        super().__init__()

        self.dataset_train = dataset_train
        self.train_collator = CoarsenLevelCollator(dataset_train)

        self.dataset_val = dataset_val
        self.val_collator = CoarsenLevelCollator(dataset_val)

        self.dataset_test = dataset_test
        self.test_collator = CoarsenLevelCollator(dataset_test)

        self.batch_size=batch_size
        self.num_workers= num_workers
        self.use_costum_ddp_sampler = use_costum_ddp_sampler


    def train_dataloader(self):
        
        if self.use_costum_ddp_sampler:
            sampler = DistributedSampler(dataset=self.dataset_train, shuffle=False)
        else:
            sampler = None
        dataloader = DataLoader(self.dataset_train, sampler=sampler, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.train_collator)

        return dataloader
    
    def val_dataloader(self):
        
        if self.use_costum_ddp_sampler:
            sampler = DistributedSampler(dataset=self.dataset_val, shuffle=False)
        else:
            sampler = None

        dataloader = DataLoader(self.dataset_val, sampler=sampler, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.val_collator)

        return dataloader

    def test_dataloader(self):

        if self.use_costum_ddp_sampler:
            sampler = DistributedSampler(dataset=self.dataset_test, shuffle=False)
        else:
            sampler = None

        dataloader = DataLoader(self.dataset_test, sampler=sampler, batch_size=self.batch_size,
                                num_workers=self.num_workers, collate_fn=self.test_collator)

        return dataloader