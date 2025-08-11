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

    def __call__(self, batch):
        # Use the default collate function to create the initial batch.
        # This will stack the tensors from __getitem__ along a new dimension.
        # The shape will be (batch_size, n, C, H, W).
        collated_batch = default_collate(batch)

        if hasattr(self.dataset, "reduce_zoom_to_batch") and self.dataset.reduce_zoom_to_batch is not None:
            assert self.dataset.reduce_zoom_to_batch > self.dataset.zoom_patch_sample
            # TODO: add reshapes for coords and rel_dists
            data_source_zooms, data_target_zooms, coords_input, coords_output, sample_dict, mask_zooms, embed_data, rel_dists_input, rel_dists_output = collated_batch
            for zoom in data_target_zooms.keys():
                b, nv, nt, n, nh = data_source_zooms[zoom].shape
                n_pix = 4 ** (int(zoom) - self.dataset.reduce_zoom_to_batch)
                n_patch = n // n_pix
                data_source_zooms[zoom] = rearrange(data_source_zooms[zoom],
                                                    "b nv nt (n_patch n_pix) nh -> (b n_patch) nv nt n_pix nh",
                                                    n_patch=n_patch, n_pix=n_pix)
                data_target_zooms[zoom] = rearrange(data_target_zooms[zoom],
                                                    "b nv nt (n_patch n_pix) nh -> (b n_patch) nv nt n_pix nh",
                                                    n_patch=n_patch, n_pix=n_pix)
                mask_zooms[zoom] = rearrange(mask_zooms[zoom],
                                             "b nv nt (n_patch n_pix) nh -> (b n_patch) nv nt n_pix nh",
                                             n_patch=n_patch, n_pix=n_pix)
                embed_data["DensityEmbedder"][0][zoom] = rearrange(embed_data["DensityEmbedder"][0][zoom],
                                                                   "b nv nt (n_patch n_pix) nh -> (b n_patch) nv nt n_pix nh",
                                                                   n_patch=n_patch, n_pix=n_pix)
            if sample_dict:
                new_sample_indices = torch.cat([torch.arange(n_patch) + (n_patch * patch_index) for patch_index in sample_dict["patch_index"]])
                sample_dict["patch_index"] = new_sample_indices
            else:
                sample_dict["patch_index"] = torch.arange(n_patch).repeat(b)
            sample_dict["zoom_patch_sample"] = torch.tensor(self.dataset.reduce_zoom_to_batch).repeat(b * n_patch)
            embed_data["VariableEmbedder"] = embed_data["VariableEmbedder"].repeat_interleave(repeats=n_patch, dim=0)
            embed_data["TimeEmbedder"] = embed_data["TimeEmbedder"].repeat_interleave(repeats=n_patch, dim=0)
            collated_batch = data_source_zooms, data_target_zooms, coords_input, coords_output, sample_dict, mask_zooms, embed_data, rel_dists_input, rel_dists_output

        if hasattr(self.dataset, "reduce_time_to_batch") and self.dataset.reduce_time_to_batch is not None:
            assert self.dataset.n_sample_timesteps % self.dataset.reduce_time_to_batch == 0
            # TODO: add reshapes for coords and rel_dists
            data_source_zooms, data_target_zooms, coords_input, coords_output, sample_dict, mask_zooms, embed_data, rel_dists_input, rel_dists_output = collated_batch
            for zoom in data_target_zooms.keys():
                b, nv, nt, n, nh = data_source_zooms[zoom].shape
                n_steps = nt // self.dataset.reduce_time_to_batch
                n_patch = self.dataset.reduce_time_to_batch
                data_source_zooms[zoom] = rearrange(data_source_zooms[zoom],
                                                    "b nv (n_patch n_steps) n nh -> (b n_patch) nv n_steps n nh",
                                                    n_patch=n_patch, n_steps=n_steps)
                data_target_zooms[zoom] = rearrange(data_target_zooms[zoom],
                                                    "b nv (n_patch n_steps) n nh -> (b n_patch) nv n_steps n nh",
                                                    n_patch=n_patch, n_steps=n_steps)
                mask_zooms[zoom] = rearrange(mask_zooms[zoom],
                                             "b nv (n_patch n_steps) n nh -> (b n_patch) nv n_steps n nh",
                                             n_patch=n_patch, n_steps=n_steps)
                embed_data["DensityEmbedder"][0][zoom] = rearrange(embed_data["DensityEmbedder"][0][zoom],
                                                                   "b nv (n_patch n_steps) n nh -> (b n_patch) nv n_steps n nh",
                                                                   n_patch=n_patch, n_steps=n_steps)
            if sample_dict:
                sample_dict["patch_index"] = sample_dict["patch_index"].repeat_interleave(repeats=n_patch, dim=0)
                sample_dict["zoom_patch_sample"] = sample_dict["zoom_patch_sample"].repeat_interleave(repeats=n_patch, dim=0)
            embed_data["VariableEmbedder"] = rearrange(embed_data["VariableEmbedder"],
                                                       "b nv (n_patch n_steps) -> (b n_patch) nv n_steps",
                                                       n_patch=n_patch, n_steps=n_steps)
            embed_data["TimeEmbedder"] = rearrange(embed_data["TimeEmbedder"],
                                                   "b (n_patch n_steps) -> (b n_patch) n_steps",
                                                   n_patch=n_patch, n_steps=n_steps)
            collated_batch = data_source_zooms, data_target_zooms, coords_input, coords_output, sample_dict, mask_zooms, embed_data, rel_dists_input, rel_dists_output
        return collated_batch


class DataModule(LightningDataModule):
    def __init__(self, dataset_train=None, dataset_val=None, dataset_test=None, batch_size=16, num_workers=16, use_costum_ddp_sampler=False):
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