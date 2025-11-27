import json
import os
from typing import Any

import time
import hydra
import netCDF4
import torch
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from omegaconf import DictConfig
from einops import rearrange
import xarray as xr

from stableclimgen.src.modules.grids.grid_utils import get_lon_lat_names
from stableclimgen.src.utils.datasets_base import BaseDataset
from stableclimgen.src.utils.file_export import save_tensor_as_netcdf, remap_healpix_to_any, export_healpix_to_netcdf
from stableclimgen.src.utils.pl_data_module import DataModule
from stableclimgen.src.utils.helpers import load_from_state_dict
import healpy as hp

import numpy as np
from scipy.interpolate import griddata


@hydra.main(version_base=None, config_path="../configs/", config_name="healpix_vae_test")
def test(cfg: DictConfig) -> None:
    """
    Main training function that initializes datasets, dataloaders, model, and trainer,
    then begins the training process.

    :param cfg: Configuration object containing all settings for training, datasets,
                model, and logging.
    """
    if not os.path.exists(os.path.dirname(cfg.output_path)):
        os.makedirs(os.path.dirname(cfg.output_path))

    test_dataset: BaseDataset = instantiate(cfg.dataloader.dataset, data_dict=cfg.data_split['test'])

    # Initialize model and trainer
    model: Any = instantiate(cfg.model)
    trainer: Trainer = instantiate(cfg.trainer)

    if cfg.ckpt_path is not None:
        model = load_from_state_dict(model, cfg.ckpt_path, print_keys=True, device=model.device)[0]

    data_module: DataModule = instantiate(cfg.dataloader.datamodule, dataset_test=test_dataset)

    # Start the training process
    start_time = time.time()
    predictions = trainer.predict(model=model, dataloaders=data_module.test_dataloader())
    end_time = time.time()

    print(f"Predicted time: {end_time - start_time:.2f} seconds")

    # get n_patches
    max_zoom = max(test_dataset.zooms)
    sampling = test_dataset.sampling_zooms_collate or test_dataset.sampling_zooms
    sampling = sampling[max_zoom]['zoom_patch_sample']
    if sampling == -1:
        n_patches = 1
    else:
        npix = hp.nside2npix(2 ** max_zoom)
        n_patches = npix // 4**(max_zoom - sampling)

    # Aggregate outputs from multiple devices
    output = torch.cat([batch["output"][max_zoom] for batch in predictions], dim=0)
    mask = torch.cat([batch["mask"][max_zoom] for batch in predictions], dim=0)

    if output.dim() == 5:
        output = rearrange(output, "(b2 b1) v t n ... -> b2 v t (b1 n) ... ", b1=n_patches)
        mask = rearrange(mask, "(b2 b1) v t n ... -> b2 v t (b1 n) ... ", b1=n_patches)
    else:
        output = rearrange(output, "(b2 b1) m v t n ... -> b2 v m t (b1 n) ... ", b1=n_patches)
        mask = rearrange(mask, "(b2 b1) m v t n ... -> b2 v m t (b1 n) ... ", b1=n_patches)

    if 'output_var' in predictions[0].keys() and predictions[0]['output_var'] is not None:
        output_var = torch.cat([batch["output_var"] for batch in predictions], dim=0)
        output_var = rearrange(output_var, "(b2 b1) v t n ... -> b2 v t (b1 n) ... ", b1=n_patches)
        
        for g, vars in enumerate(test_dataset.variables):
            for k,var in enumerate(vars):
                output_var[:,g,...,k]= test_dataset.var_normalizers[max(test_dataset.var_normalizers.keys())][var].denormalize_var(output_var[:, g,...,k], data=output[:, g,...,k])

        output_var = dict(zip(test_dataset.var_groups, output_var.split(1, dim=-1)))
        torch.save(output_var, cfg.output_path.replace(".pt", "_var.pt"))

    for g, vars in enumerate(test_dataset.variables):
        for k,var in enumerate(vars):
            output[:,g,...,k]= test_dataset.var_normalizers[max(test_dataset.var_normalizers.keys())][var].denormalize(output[:, g,...,k])

    output = dict(zip(test_dataset.var_groups, output.split(1, dim=1)))
    torch.save(output, cfg.output_path)
    mask = dict(zip(test_dataset.var_groups, (1-mask).split(1, dim=1)))
    torch.save(mask, cfg.output_path.replace(".pt", "_mask.pt"))

    if hasattr(cfg, "save_netcdf") and cfg.save_netcdf:
        export_healpix_to_netcdf(test_dataset, max_zoom, output, mask,
                                 cfg.logger.run_id if "MLFlowLogger" in cfg.logger['_target_'] else cfg.logger.id,
                                 cfg.output_path)

if __name__ == "__main__":
    test()
