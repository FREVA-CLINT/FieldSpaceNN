import json
import os
from typing import Any

import hydra
import numpy as np
import torch
import xarray as xr
from hydra.utils import instantiate
from omegaconf import DictConfig

from stableclimgen.src.utils.grid_utils_healpix import healpix_pixel_lonlat_torch, get_mapping_to_healpix_grid
from stableclimgen.src.utils import normalizer as normalizers
from stableclimgen.src.utils.grid_utils_icon import get_coords_as_tensor

torch.manual_seed(42)


def get_data(ds, ts, variables): 

    ds = ds.isel(time=ts)

    data_g = []
    for variable in variables:
        data = torch.tensor(ds[variable].values)
        data = data[0][0] if data.dim() > 2  else data
        data = data[0] if data.dim() > 1  else data
        data_g.append(data)

    data_g = torch.stack(data_g, dim=-1)

    ds.close()

    return data_g


@hydra.main(version_base=None, config_path="../configs/", config_name="healpix_vae_test")
def test(cfg: DictConfig) -> None:
    """
    Main training function that initializes datasets, dataloaders, model, and trainer,
    then begins the training process.

    :param cfg: Configuration object containing all settings for training, datasets,
                model, and logging.
    """
    with open(cfg.dataloader.dataset.data_dict) as json_file:
        data_dict = json.load(json_file)

    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)

    variables_train = data_dict["train"]['source']["variables"]
    variables = data_dict["test"]["variables"]
    files = data_dict["test"]["files"]
    coarsen_level_batches = cfg.coarsen_level_batches if "coarsen_level_batches" in cfg.keys() else -1

    model: Any = instantiate(cfg.model)
    ckpt = torch.load(cfg.ckpt_path)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    
    with open(cfg.dataloader.dataset.norm_dict) as json_file:
        norm_dict = json.load(json_file)

    var_normalizers = {}
    for var in variables:
        norm_class = norm_dict[var]['normalizer']['class']
        assert norm_class in normalizers.__dict__.keys(), f'normalizer class {norm_class} not defined'

        var_normalizers[var] = normalizers.__getattribute__(norm_class)(
            norm_dict[var]['stats'],
            norm_dict[var]['normalizer'])

    processing_nside = cfg.dataloader.dataset.processing_nside
    in_nside = cfg.dataloader.dataset.in_nside
    coords_processing = healpix_pixel_lonlat_torch(processing_nside)

    if processing_nside == in_nside:
        input_mapping = np.arange(coords_processing.shape[0])[:, np.newaxis]
        input_in_range = np.ones_like(input_mapping, dtype=bool)[:, np.newaxis]
        input_coordinates = None
    else:
        if in_nside is not None:
            input_coordinates = healpix_pixel_lonlat_torch(in_nside)
        else:
            assert (cfg.dataloader.dataset.in_grid is not None)
            input_coordinates = get_coords_as_tensor(xr.open_zarr(cfg.dataloader.dataset.in_grid, decode_times=False),
                                                     grid_type='cell')  # change for other grids
        mapping = get_mapping_to_healpix_grid(coords_processing,
                                              input_coordinates,
                                              search_radius=cfg.dataloader.dataset.search_radius,
                                              max_nh=cfg.dataloader.dataset.nh_input,
                                              lowest_level=0,
                                              periodic_fov=None)

        input_mapping = mapping["indices"]
        input_in_range = mapping["in_rng_mask"]

    var_indices = [np.where(var==np.array(variables_train))[0][0] for var in variables]

    input_coordinates = input_coordinates[input_mapping].unsqueeze(dim=0) if input_coordinates is not None else input_coordinates

    for file_idx, file in enumerate(files):
        ds = xr.open_zarr(file, decode_times=False)
        for tp in data_dict["test"]['timepoints']:
            data = get_data(ds, ts=data_dict["test"]['timepoints'], variables=variables)
            data = data[input_mapping]
            for k, var in enumerate(variables):
                data[:,:,k] = var_normalizers[var].normalize(data[:,:,k])

            embed_data = {'VariableEmbedder': torch.tensor(var_indices).unsqueeze(dim=0)}

            if coarsen_level_batches!=-1:
                indices_sample = {'sample': torch.arange(data.shape[0]//4**coarsen_level_batches),
                    'sample_level': torch.ones(data.shape[0]//4**coarsen_level_batches, dtype=int).view(-1)*coarsen_level_batches}
                
                data = data.view(len(indices_sample['sample']),-1,data.shape[1],data.shape[-1],1)
                if input_coordinates is not None:
                    input_coordinates = input_coordinates.view(len(indices_sample['sample']),-1,input_coordinates.shape[2],2)

                embed_data['VariableEmbedder'] = embed_data['VariableEmbedder'].repeat_interleave(data.shape[0],dim=0)
            else:
                data = torch.tensor(data[np.newaxis,...,np.newaxis])
                indices_sample=None
            
            mask = torch.zeros(*data.shape)

            with torch.no_grad():
                output, _ = model(data, coords_input=input_coordinates, coords_output=None, indices_sample=indices_sample, mask=mask, emb=embed_data)

            output = output.view(-1,output.shape[-1])
            for k, var in enumerate(variables):
                output[:,k] = var_normalizers[var].denormalize(output[:,k])
            
            output = dict(zip(variables, output.split(1, dim=-1)))

            torch.save(output, os.path.join(cfg.output_dir,os.path.basename(file).replace('.zarr',f'_{tp}.pt')))

if __name__ == "__main__":
    test()
