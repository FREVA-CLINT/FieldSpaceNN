import json
import os
from typing import Any

import numpy as np
import hydra
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf
from .utils.pl_data_module import DataModule
import torch
from .utils import normalizer as normalizers
from .modules.grids.grid_utils_icon import get_nh_variable_mapping_icon
from .modules.grids.grid_utils import get_coords_as_tensor

import xarray as xr

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


@hydra.main(version_base=None, config_path="/Users/maxwitte/work/stableclimgen/stableclimgen/configs", config_name="mgno_transformer_test")
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
    files = np.loadtxt(data_dict["test"]["files"], dtype='str')
    grid_input = data_dict["test"]["grid"]
    coarsen_level_batches = cfg.coarsen_level_batches if "coarsen_level_batches" in cfg.keys() else -1
    p_dropout = cfg.p_dropout if "p_dropout" in cfg.keys() else 0
    drop_vars = cfg.drop_vars if "drop_vars" in cfg.keys() else False

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

    grid_processing = cfg.mgrids.grid_file
    coords_processing = get_coords_as_tensor(xr.open_dataset(grid_processing), lon='clon', lat='clat', target='numpy')
    
    if  grid_input != grid_processing:
        input_mapping, input_in_range, positions = get_nh_variable_mapping_icon(grid_processing, 
                                                                    ['cell'], 
                                                                    grid_input, 
                                                                    ['cell'], 
                                                                    search_radius=cfg.dataloader.dataset.search_radius, 
                                                                    max_nh=cfg.dataloader.dataset.nh_input,
                                                                    lowest_level=0,
                                                                    coords_icon=coords_processing,
                                                                    scale_input=1.,
                                                                    periodic_fov= None)
                        
        input_mapping = input_mapping['cell']['cell']
        input_in_range = input_in_range['cell']['cell']
        positions = positions['cell']['cell']
        input_coordinates = get_coords_as_tensor(xr.open_dataset(grid_input), lon='clon', lat='clat', target='torch')
    else:
        input_mapping = torch.arange(coords_processing.shape[0]).unsqueeze(dim=-1)
        input_in_range = torch.ones_like(input_mapping, dtype=bool).unsqueeze(dim=-1)
        input_coordinates = None
        input_in_range = None

    var_indices = [np.where(var==np.array(variables_train))[0][0] for var in variables]

    input_coordinates = input_coordinates[input_mapping].unsqueeze(dim=0) if input_coordinates is not None else input_coordinates

    for file_idx, file in enumerate(files):
        ds = xr.open_dataset(file)
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
            
            mask = torch.zeros(*data.shape, dtype=bool)

            if input_in_range is not None:
                if indices_sample is not None:
                    input_in_range = input_in_range.view(len(indices_sample['sample']),-1,input_in_range.shape[-1])
                else:
                    input_in_range = input_in_range.view(1,-1)
                mask[input_in_range==False] = True

            b,n,nh,nv = data.shape[:4]
            if p_dropout > 0 and not drop_vars:
                drop_mask_p = (torch.rand((b,n,nh))<p_dropout).bool()
                mask[drop_mask_p]=True

            elif drop_vars:
                drop_mask_p = (torch.rand((b,n,nh,nv))<p_dropout).bool()
                mask[drop_mask_p]=True
            
            data[mask]=0

            with torch.no_grad():
                output = model(data, coords_input=input_coordinates, coords_output=None, indices_sample=indices_sample, mask=mask, emb=embed_data)

            output = output.view(-1,output.shape[-1])
            for k, var in enumerate(variables):
                output[:,k] = var_normalizers[var].denormalize(output[:,k])
            
            output = dict(zip(variables, output.split(1, dim=-1)))

            torch.save(output, os.path.join(cfg.output_dir,os.path.basename(file).replace('.nc',f'_{tp}.pt')))
            torch.save(mask, os.path.join(cfg.output_dir,os.path.basename(file).replace('.nc',f'_{tp}_mask.pt')))

if __name__ == "__main__":
    test()
