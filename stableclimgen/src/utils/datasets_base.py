import copy
import json

import numpy as np
import torch
import xarray as xr
from omegaconf import ListConfig
from torch.utils.data import Dataset

from . import normalizer as normalizers

def skewed_random_p(size, exponent=2, max_p=0.9):
    uniform_random = torch.rand(size)
    skewed_random =  max_p*(1 - uniform_random ** exponent)
    return skewed_random

class BaseDataset(Dataset):
    def __init__(self,
                 n_sample_patches,
                 data_dict,
                 norm_dict=None,
                 variables_source_train=None,
                 variables_target_train=None,
                 lazy_load=True,
                 p_dropout=0,
                 p_patchify=0,
                 p_patch_dropout=0,
                 zoom_patchify_max=0,
                 p_drop_vars=0,
                 n_drop_vars=-1,
                 p_drop_timesteps=-1,
                 random_p = False,
                 skewness_exp = 2,
                 n_sample_vars=-1,
                 n_sample_timesteps=1,
                 deterministic=False,
                 ):
        
        super(BaseDataset, self).__init__()
        
        self.n_sample_patches = n_sample_patches
        self.norm_dict = norm_dict
        self.lazy_load = lazy_load
        self.random_p = random_p
        self.p_dropout = p_dropout
        self.skewness_exp = skewness_exp
        self.n_sample_vars = n_sample_vars
        self.n_sample_timesteps = n_sample_timesteps
        self.deterministic = deterministic
        self.n_drop_vars = n_drop_vars
        self.p_drop_vars = p_drop_vars
        self.p_drop_timesteps = p_drop_timesteps

        self.variables_source = data_dict["source"]["variables"]
        self.variables_target = data_dict["target"]["variables"]

        self.variables_source_train = variables_source_train if variables_source_train is not None else self.variables_source
        self.variables_target_train = variables_target_train if variables_target_train is not None else self.variables_target

        if isinstance(data_dict["source"]["files"], list) or isinstance(data_dict["source"]["files"], ListConfig):
            self.files_source = np.array(data_dict["source"]["files"])
            self.files_target = np.array(data_dict["target"]["files"])
        else:
            self.files_source = np.loadtxt(data_dict["source"]["files"], dtype='str', ndmin=1)
            self.files_target = np.loadtxt(data_dict["target"]["files"], dtype='str', ndmin=1)

        if "timesteps" in data_dict.keys():
            self.sample_timesteps = []
            for t in data_dict["timesteps"]:
                if isinstance(t, int) or "-" not in t:
                    self.sample_timesteps.append(int(t))
                else:
                    start, end = map(int, t.split("-"))
                    self.sample_timesteps += list(range(start, end))
            self.sample_timesteps = self.sample_timesteps
        else:
            self.sample_timesteps = None


        self.time_steps_files = []
        for k, file in enumerate(self.files_source):
            ds = xr.open_dataset(file)
            self.time_steps_files.append(len(ds.time))

        self.index_map = []
        time_idx_global = 0
        for file_idx, num_timesteps in enumerate(self.time_steps_files):
            for time_idx in range(num_timesteps):
                if self.sample_timesteps is None or time_idx_global in self.sample_timesteps:
                    for region_idx in range(self.n_sample_patches):
                        self.index_map.append((file_idx, time_idx, region_idx))
                
                time_idx_global += 1

        self.index_map = np.array(self.index_map)

        with open(norm_dict) as json_file:
            norm_dict = json.load(json_file)

        self.var_normalizers = {}
        for var in self.variables_source:
            norm_class = norm_dict[var]['normalizer']['class']
            assert norm_class in normalizers.__dict__.keys(), f'normalizer class {norm_class} not defined'

            self.var_normalizers[var] = normalizers.__getattribute__(norm_class)(
                norm_dict[var]['stats'],
                norm_dict[var]['normalizer'])

        self.len_dataset = self.index_map.shape[0] - (self.n_sample_timesteps - 1)
    
    def get_indices_from_patch_idx(self, patch_idx):
        raise NotImplementedError

    def get_files(self, file_path_source, file_path_target=None):
      
        if self.lazy_load:
            ds_source = xr.open_dataset(file_path_source, decode_times=False)
        else:
            ds_source = xr.load_dataset(file_path_source, decode_times=False)

        if file_path_target is None:
            ds_target = copy.deepcopy(ds_source)

        elif file_path_target==file_path_source:
            ds_target = ds_source
            
        else:
            if self.lazy_load:
                ds_target = xr.open_dataset(file_path_target, decode_times=False)
            else:
                ds_target = xr.load_dataset(file_path_target, decode_times=False)

        return ds_source, ds_target


    def get_data(self, ds, time_idx, patch_idx, variables):
        
        patch_indices = self.get_indices_from_patch_idx(patch_idx)

        assert patch_indices.ndim == 2,"patch indices must have two dimensions"

        n, nh = patch_indices.shape

        nt = self.n_sample_timesteps
        patch_indices = patch_indices.reshape(-1)

        time_indices = torch.arange(time_idx, time_idx + nt, 1)

        isel_dict = {"time": time_indices}

        if 'ncells' in dict(ds.sizes).keys():
            isel_dict["ncells"] = patch_indices

        elif 'cell' in dict(ds.sizes).keys():
            isel_dict["cell"] = patch_indices

        elif 'cells' in dict(ds.sizes).keys():
            isel_dict["cells"] = patch_indices

        else:
            raise ValueError

        ds = ds.isel(isel_dict)

        data_g = []
        for variable in variables:
            data = torch.tensor(ds[variable].values)
            data = data.view(nt, -1,n,nh)
            data_g.append(data.transpose(0,1))

        data_g = torch.stack(data_g, dim=0)
        data_g = data_g.view(len(variables),nt,n,nh,-1)

        data_t = torch.tensor(ds["time"].values)

        ds.close()

        return data_g, data_t

    def __getitem__(self, index):
        
        file_index, time_index, patch_index = self.index_map[index]

        source_file = self.files_source[file_index]
        target_file = self.files_target[file_index]

        ds_source, ds_target = self.get_files(source_file, file_path_target=target_file)

        if self.n_sample_vars != -1 and self.n_sample_vars != len(self.variables_source):
            sample_vars = torch.randperm(len(self.variables_source))[:self.n_sample_vars]
        else:
            sample_vars = torch.arange(len(self.variables_source))

        variables_source = np.array([self.variables_source[i.item()] for i in sample_vars])
        variables_target = np.array([self.variables_target[i.item()] for i in sample_vars])

        data_source, time_source = self.get_data(ds_source, time_index, patch_index, variables_source)

        if ds_target is not None:
            data_target, time_target = self.get_data(ds_target, time_index, patch_index, variables_source)
        else:
            data_target, time_target = data_source, time_source

        nv, nt, n, nh, f = data_source.shape

        drop_vars = torch.rand(1) < self.p_drop_vars
        drop_timesteps = torch.rand(1) < self.p_drop_timesteps
        drop_mask = torch.zeros((nv, nt, n, nh), dtype=bool)

        if self.random_p and drop_vars:
            p_dropout = skewed_random_p(nv, exponent=self.skewness_exp, max_p=self.p_dropout)
        elif self.random_p:
            p_dropout = skewed_random_p(1, exponent=self.skewness_exp, max_p=self.p_dropout)
        else:
            p_dropout = torch.tensor(self.p_dropout)

        if self.p_dropout > 0 and not drop_vars and not drop_timesteps:
            drop_mask_p = (torch.rand((nt, n, nh)) < p_dropout).bool()
            drop_mask[:,drop_mask_p] = True

        elif self.p_dropout > 0 and drop_vars:
            drop_mask_p = (torch.rand((nv, nt, n, nh)) < p_dropout).bool()
            drop_mask[drop_mask_p] = True

        elif self.p_dropout > 0 and drop_timesteps:
            drop_mask_p = (torch.rand(nt)<self.p_dropout).bool()
            drop_mask[:,drop_mask_p]=True

        if self.n_drop_vars!=-1 and self.n_drop_vars < nv:
            not_drop_vars = torch.randperm(nv)[:(nv-self.n_drop_vars)]
            drop_mask[not_drop_vars] = (drop_mask[not_drop_vars]*0).bool()
    
        for k, var in enumerate(variables_source):
            data_source[k] = self.var_normalizers[var].normalize(data_source[k])
        
        for k, var in enumerate(variables_target):
            data_target[k] = self.var_normalizers[var].normalize(data_target[k])

        data_source[drop_mask] = 0

        if self.zoom_patch_sample == -1:
            sample_dict = {
                'zoom': self.zoom
                }
        else:
            sample_dict = {
                'patch_index': patch_index,
                'zoom_patch_sample': self.zoom_patch_sample,
                'zoom': self.zoom
                }

        embed_data = {'VariableEmbedder': sample_vars.unsqueeze(-1).repeat(1,nt),
                      'TimeEmbedder': time_source.float()}
        
        coords_input = torch.tensor([])
        coords_output = torch.tensor([])
        rel_dists_input = torch.tensor([])
        rel_dists_output = torch.tensor([])

        return data_source.float(), data_target.float(), coords_input, coords_output, sample_dict, drop_mask, embed_data, rel_dists_input, rel_dists_output

    def __len__(self):
        return self.len_dataset