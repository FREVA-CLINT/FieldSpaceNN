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
from stableclimgen.src.utils.file_export import save_tensor_as_netcdf
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

    target_variables = []
    for grid_type, variables_grid_type in test_dataset.grid_types_vars.items():
        target_variables += list(set(variables_grid_type))

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

        for k, var in enumerate(target_variables):
            output_var[:, k] = test_dataset.var_normalizers[var].denormalize_var(output_var[:, k], data=output[:, k])

        output_var = dict(zip(target_variables, output_var.split(1, dim=-1)))
        torch.save(output_var, cfg.output_path.replace(".pt", "_var.pt"))

    for k, var in enumerate(target_variables):
        output[:,k] = test_dataset.var_normalizers[var].denormalize(output[:, k])

    output = dict(zip(target_variables, output.split(1, dim=1)))
    torch.save(output, cfg.output_path)
    mask = dict(zip(target_variables, mask.split(1, dim=1)))
    torch.save(mask, cfg.output_path.replace(".pt", "_mask.pt"))

    if hasattr(cfg, "save_netcdf") and cfg.save_netcdf:
        output_remapped = {}
        # get mapping
        for grid_type, variables_grid_type in test_dataset.grid_types_vars.items():
            variables = set(variables_grid_type)
            indices = test_dataset.mapping[max_zoom][grid_type]['indices'][...,[0]]
            lat_dim, lon_dim = 36, 72 # TODO: get dimensions from nc dataset
            for var in variables:
                s = 3
                leading_dims = output[var].shape[:s]
                trailing_dims = output[var].shape[s + 1:]

                output_flat_shape = leading_dims + (lat_dim * lon_dim,) + trailing_dims
                output_rem = torch.full(output_flat_shape, float('nan'), dtype=output[var].dtype,
                                        device=output[var].device)

                map_shape = [1] * output[var].dim()
                map_shape[s] = output[var].shape[s]
                map_reshaped = indices.view(map_shape)

                expanded_map = map_reshaped.expand(output[var].shape)
                output_rem.scatter_(s, expanded_map, output[var])

                output_grid_nan = output_rem.reshape(leading_dims + (lat_dim, lon_dim) + trailing_dims)

                tensor_np = output_grid_nan.cpu().numpy()
                interpolated_tensor_np = np.copy(tensor_np)

                it = np.nditer(tensor_np[..., 0, 0, :], flags=['multi_index'], op_flags=['readonly'])

                while not it.finished:
                    idx = it.multi_index
                    grid_2d = tensor_np[idx[:s] + (slice(None), slice(None)) + idx[s:]]

                    if np.isnan(grid_2d).any():
                        y, x = np.mgrid[0:lat_dim, 0:lon_dim]
                        valid_mask = ~np.isnan(grid_2d)
                        if not np.any(valid_mask):
                            it.iternext()
                            continue

                        points = np.array([y[valid_mask], x[valid_mask]]).T
                        values = grid_2d[valid_mask]

                        interpolated_grid = griddata(points, values, (y, x), method='nearest')
                        interpolated_tensor_np[idx[:s] + (slice(None), slice(None)) + idx[s:]] = interpolated_grid

                    it.iternext()

                output_remapped[var] = torch.from_numpy(interpolated_tensor_np).to(output[var].device).reshape(
                    output_flat_shape)


        reference_nc_path = test_dataset.data_dict['target'][max_zoom]['files'][0]

        if test_dataset.sample_timesteps:
            timesteps = xr.open_dataset(reference_nc_path, decode_times=False)["time"][test_dataset.sample_timesteps]
        else:
            timesteps = xr.open_dataset(reference_nc_path, decode_times=False)["time"][:]

        if ".nc" in reference_nc_path:
            for grid_type, variables_grid_type in test_dataset.grid_types_vars.items():
                lon, lat = get_lon_lat_names(grid_type)
                dimensions = ('time', lat, lon)
                save_tensor_as_netcdf({key: value for key, value in output_remapped.items() if key in variables_grid_type},
                                      timesteps,
                                      cfg.output_path.replace(".pt", ".nc"),
                                      dimensions,
                                      reference_nc_path)

if __name__ == "__main__":
    test()
