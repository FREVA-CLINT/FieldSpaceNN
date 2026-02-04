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

    def _reshape_group(self, data_source_zooms, data_target_zooms, emb_data, sample_configs):
        if not data_source_zooms:
            return data_source_zooms, data_target_zooms, emb_data, sample_configs

        base_configs = self.dataset.sampling_zooms_collate or self.dataset.sampling_zooms
        cleaned_configs = {}
        for zoom, conf in base_configs.items():
            if zoom in sample_configs:
                conf_copy = copy.deepcopy(conf)
                conf_copy["patch_index"] = sample_configs[zoom].get("patch_index")
                cleaned_configs[zoom] = conf_copy
        sample_configs = cleaned_configs

        n_patch = None
        for zoom in data_target_zooms.keys():
            if self.dataset.sampling_zooms_collate[zoom]["zoom_patch_sample"] > self.dataset.sampling_zooms[zoom]["zoom_patch_sample"]:
                b, nv, nt, n, d, f = data_source_zooms[zoom].shape
                n_pix = 4 ** (int(zoom) - self.dataset.sampling_zooms_collate[zoom]["zoom_patch_sample"])
                n_patch = n // n_pix
                data_source_zooms[zoom] = rearrange(data_source_zooms[zoom],
                                                    "b nv nt (n_patch n_pix) d f  -> (b n_patch) nv nt n_pix d f ",
                                                    n_patch=n_patch, n_pix=n_pix)
                data_target_zooms[zoom] = rearrange(data_target_zooms[zoom],
                                                    "b nv nt (n_patch n_pix) d f  -> (b n_patch) nv nt n_pix d f ",
                                                    n_patch=n_patch, n_pix=n_pix)

                patch_index = sample_configs[zoom]["patch_index"]
                new_patch_indices = torch.cat([torch.arange(n_patch) + (n_patch * patch_idx) for patch_idx in patch_index])
                sample_configs[zoom]["patch_index"] = new_patch_indices

                emb_data["DensityEmbedder"][0][zoom] = rearrange(emb_data["DensityEmbedder"][0][zoom],
                                                                 "b nv nt (n_patch n_pix) d f  -> (b n_patch) nv nt n_pix d f ",
                                                                 n_patch=n_patch, n_pix=n_pix)
                emb_data["TimeEmbedder"][zoom] = emb_data["TimeEmbedder"][zoom].repeat_interleave(repeats=n_patch, dim=0)
        if n_patch is not None:
            emb_data["VariableEmbedder"] = emb_data["VariableEmbedder"].repeat_interleave(repeats=n_patch, dim=0)
            emb_data["DensityEmbedder"] = (emb_data["DensityEmbedder"][0], emb_data["VariableEmbedder"])

        n_patch = None
        for zoom in data_target_zooms.keys():
            if (self.dataset.sampling_zooms_collate[zoom]["n_past_ts"] < self.dataset.sampling_zooms[zoom]["n_past_ts"]
             or self.dataset.sampling_zooms_collate[zoom]["n_future_ts"] < self.dataset.sampling_zooms[zoom]["n_future_ts"]):
                b, nv, nt, n, d, f = data_source_zooms[zoom].shape
                n_steps = self.dataset.sampling_zooms_collate[zoom]["n_past_ts"] + self.dataset.sampling_zooms_collate[zoom]["n_future_ts"] + 1
                n_patch = nt // n_steps
                data_source_zooms[zoom] = rearrange(data_source_zooms[zoom],
                                                    "b nv (n_patch n_steps) n d f -> (b n_patch) nv n_steps n d f",
                                                    n_patch=n_patch, n_steps=n_steps)
                data_target_zooms[zoom] = rearrange(data_target_zooms[zoom],
                                                    "b nv (n_patch n_steps) n d f -> (b n_patch) nv n_steps n d f",
                                                    n_patch=n_patch, n_steps=n_steps)

                sample_configs[zoom]["patch_index"] = sample_configs[zoom]["patch_index"].repeat_interleave(repeats=n_patch, dim=0)

        for zoom in emb_data["DensityEmbedder"][0].keys():
            if (self.dataset.sampling_zooms_collate[zoom]["n_past_ts"] < self.dataset.sampling_zooms[zoom]["n_past_ts"]
             or self.dataset.sampling_zooms_collate[zoom]["n_future_ts"] < self.dataset.sampling_zooms[zoom]["n_future_ts"]):
                n_steps = self.dataset.sampling_zooms_collate[zoom]["n_past_ts"] + self.dataset.sampling_zooms_collate[zoom]["n_future_ts"] + 1
                n_patch = emb_data["DensityEmbedder"][0][zoom].shape[2] // n_steps
                emb_data["DensityEmbedder"][0][zoom] = rearrange(emb_data["DensityEmbedder"][0][zoom],
                                                                "b nv (n_patch n_steps) n d f -> (b n_patch) nv n_steps n d f",
                                                                n_patch=n_patch, n_steps=n_steps)
                emb_data["TimeEmbedder"][zoom] = rearrange(emb_data["TimeEmbedder"][zoom],
                                                          "b (n_patch n_steps) -> (b n_patch) n_steps",
                                                          n_patch=n_patch, n_steps=n_steps)

        if n_patch is not None:
            emb_data["VariableEmbedder"] = emb_data["VariableEmbedder"].repeat_interleave(repeats=n_patch, dim=0)
            emb_data["DensityEmbedder"] = (emb_data["DensityEmbedder"][0], emb_data["VariableEmbedder"])

        return data_source_zooms, data_target_zooms, emb_data, sample_configs

    def __call__(self, batch):
        # Use the default collate function to create the initial batch.
        # This will stack the tensors from __getitem__ along a new dimension.
        # The shape will be (batch_size, n, C, H, W).

        if hasattr(self.dataset, "sampling_zooms_collate") and self.dataset.sampling_zooms_collate is not None:
            (source_zooms_2d, source_zooms_3d,
             target_zooms_2d, target_zooms_3d,
             emb_2d, emb_3d,
             sample_configs_2d, sample_configs_3d) = default_collate(batch)

            source_zooms_2d, target_zooms_2d, emb_2d, sample_configs_2d = self._reshape_group(
                source_zooms_2d, target_zooms_2d, emb_2d, sample_configs_2d
            )
            source_zooms_3d, target_zooms_3d, emb_3d, sample_configs_3d = self._reshape_group(
                source_zooms_3d, target_zooms_3d, emb_3d, sample_configs_3d
            )

            return source_zooms_2d, source_zooms_3d, target_zooms_2d, target_zooms_3d, emb_2d, emb_3d, sample_configs_2d, sample_configs_3d
        else:
            return default_collate(batch)


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
