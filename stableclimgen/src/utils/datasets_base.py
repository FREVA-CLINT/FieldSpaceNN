import copy
import json
from collections import defaultdict

import numpy as np
import torch
import xarray as xr
from omegaconf import ListConfig
from torch.utils.data import Dataset
from ..modules.grids.grid_utils import get_coords_as_tensor,get_grid_type_from_var,get_mapping_weights,encode_zooms

import warnings
warnings.filterwarnings("ignore", message=".*fails while guessing")

from . import normalizer as normalizers

def skewed_random_p(size, exponent=2, max_p=0.9):
    uniform_random = torch.rand(size)
    skewed_random =  max_p*(1 - uniform_random ** exponent)
    return skewed_random


#def create_mask(random_p, drop_mask, ):

class BaseDataset(Dataset):
    def __init__(self,
                 n_sample_patches,
                 data_dict,
                 mapping_fcn=None,
                 norm_dict=None,
                 variables_source_train=None,
                 variables_target_train=None,
                 lazy_load=True,
                 p_dropout=0,
                 p_patchify=0,
                 p_patch_dropout=0,
                 zoom_patchify_max=0,
                 mask_zooms=None,
                 p_drop_vars=0,
                 n_drop_vars=-1,
                 p_drop_timesteps=-1,
                 random_p = False,
                 skewness_exp = 2,
                 n_sample_vars=-1,
                 n_sample_timesteps=1,
                 deterministic=False,
                 out_zooms=[],
                 output_binary_mask=False,
                 output_differences=True
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
        self.out_zooms = out_zooms
        self.output_differences = output_differences
        self.output_binary_mask = output_binary_mask
        self.mask_zooms = mask_zooms

        self.variables_source = data_dict["source"]["variables"]
        self.variables_target = data_dict["target"]["variables"]

        #TODO fix the variable ids if multi var training/inference
        self.variables_source_train = variables_source_train if variables_source_train is not None else self.variables_source
        self.variables_target_train = variables_target_train if variables_target_train is not None else self.variables_target

        self.var_indices = dict(zip(self.variables_source_train, np.arange(len(self.variables_source_train))))

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

         #  mapping = mapping_fcn()
        grid_types = [get_grid_type_from_var(ds, var) for var in self.variables_source]
        self.vars_grid_types = dict(zip(self.variables_source, grid_types))
        self.grid_types = np.unique(grid_types)

        self.grid_types_vars = defaultdict(list)

        for var, gtype in zip(self.variables_source, grid_types):
            self.grid_types_vars[gtype].append(var)

        coords = [get_coords_as_tensor(ds, grid_type=grid_type) for grid_type in self.grid_types]
        self.mapping = dict(zip(self.grid_types, [mapping_fcn(coords_, max(out_zooms)) for coords_ in coords]))

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
            ds_target = copy.deepcopy(ds_source)
            
        else:
            if self.lazy_load:
                ds_target = xr.open_dataset(file_path_target, decode_times=False)
            else:
                ds_target = xr.load_dataset(file_path_target, decode_times=False)

        return ds_source, ds_target

    #def map_data(self):


    def get_data(self, ds, time_idx, patch_idx, variables, mapping, n_avg=1, mask_mapping=None, drop_mask_raw=None):
        # Fetch raw patch indices

        # keep nearest neighbour only
        patch_indices = (mapping['indices'][self.get_indices_from_patch_idx(patch_idx)])[...,[0]]  # shape: (n, 1)
        
        n,nh = patch_indices.shape

        #out shape: nv, nt, n, 1, nh
        if drop_mask_raw is not None:
            drop_mask_raw = drop_mask_raw[:,:,patch_indices]
            mask = ((drop_mask_raw==False) * mask_mapping[self.get_indices_from_patch_idx(patch_idx)][...,0].view(1,1,-1,1)).bool()
        elif mask_mapping is not None:
            mask = mask_mapping[self.get_indices_from_patch_idx(patch_idx)][...,0].view(1,1,-1,1)
        else:
            mask = None   
        
        nt = self.n_sample_timesteps

        time_indices = torch.arange(time_idx, time_idx + nt, 1)

        isel_dict = {"time": time_indices}
        patch_dim = [d for d in ds.dims if "cell" in d or "ncells" in d][0]
        isel_dict[patch_dim] = patch_indices.view(-1)

        ds = ds.isel(isel_dict)

        data_g = []
        for i, variable in enumerate(variables):
            data = torch.tensor(ds[variable].values)  # shape: (nt, n)
            data = data.view(nt, n, nh)
            data_g.append(data)  # (1, nt, n)

        data_g = torch.stack(data_g, dim=0)  # (nv, nt, n)
        
        data_t = torch.tensor(ds["time"].values)
        ds.close()

        if mask is not None:
            mask = mask == False
            data_g[mask] = 0

        return data_g, data_t, mask


    def get_mask(self, grid_type, nv):
        nt = self.n_sample_timesteps
        n = self.mapping[grid_type][max(self.out_zooms)]['indices'].max() +1

        drop_vars = torch.rand(1) < self.p_drop_vars
        drop_timesteps = torch.rand(1) < self.p_drop_timesteps
        drop_mask = torch.zeros((nv, nt, n), dtype=bool)

        if self.random_p and drop_vars:
            p_dropout = skewed_random_p(nv, exponent=self.skewness_exp, max_p=self.p_dropout)
        elif self.random_p:
            p_dropout = skewed_random_p(1, exponent=self.skewness_exp, max_p=self.p_dropout)
        else:
            p_dropout = torch.tensor(self.p_dropout)

        if self.p_dropout > 0 and not drop_vars and not drop_timesteps:
            drop_mask_p = (torch.rand((nt, n)) < p_dropout).bool()
            drop_mask[:,drop_mask_p] = True

        elif self.p_dropout > 0 and drop_vars:
            drop_mask_p = (torch.rand((nv, nt, n)) < p_dropout.view(-1,1,1)).bool()
            drop_mask[drop_mask_p] = True

        elif self.p_dropout > 0 and drop_timesteps:
            drop_mask_p = (torch.rand(nt)<self.p_dropout).bool()
            drop_mask[:,drop_mask_p]=True

        if self.n_drop_vars!=-1 and self.n_drop_vars < nv:
            not_drop_vars = torch.randperm(nv)[:(nv-self.n_drop_vars)]
            drop_mask[not_drop_vars] = (drop_mask[not_drop_vars]*0).bool()

        return drop_mask
  
    

    def __getitem__(self, index):
        
        file_index, time_index, patch_index = self.index_map[index]

        source_file = self.files_source[file_index]
        target_file = self.files_target[file_index]

        ds_source, ds_target = self.get_files(source_file, file_path_target=target_file)

        # get variables to sample
        if self.n_sample_vars != -1 and self.n_sample_vars != len(self.variables_source):
            sample_vars = torch.randperm(len(self.variables_source))[:self.n_sample_vars]
        else:
            sample_vars = torch.arange(len(self.variables_source))

        variables_sample = np.array([self.variables_source_train[i.item()] for i in sample_vars])
                
        #get random dropout mask
        

        data_source = [] #dict(zip(self.out_zooms, [[] for _ in self.out_zooms]))
        masks = [] #dict(zip(self.out_zooms, [[] for _ in self.out_zooms]))

        max_zoom = max(self.out_zooms)
        var_indices = []

        k = 0
        variables_ordered = []
        for grid_type, variables in self.grid_types_vars.items():
            variables = [var for var in variables if var in variables_sample]
            variables_ordered += variables
            mapping = self.mapping[grid_type]
            mask = get_mapping_weights(mapping)[max_zoom]
            drop_mask = self.get_mask(grid_type, len(variables))

            data_source_g, time_source, mask = self.get_data(ds_source, time_index, patch_index, variables, mapping[max_zoom], mask_mapping=mask, drop_mask_raw=drop_mask)
            masks.append(mask)
            data_source.append(data_source_g)
            var_indices.append([self.var_indices[var] for var in variables])

        var_indices = np.concatenate(var_indices,axis=0)
        data_source = torch.concat(data_source, dim=0)
        masks = torch.concat(masks, dim=0)

        data_target = []
        if ds_target is None:
            data_target = copy.deepcopy(data_source)
        else:
            data_target = []
            for grid_type, variables in self.grid_types_vars.items():
                variables = [var for var in variables if var in variables_sample]
                mapping = self.mapping[grid_type]
                data_target_g, _, _ = self.get_data(ds_target, time_index, patch_index, variables, mapping[max_zoom])
                data_target.append(data_target_g)

            data_target = torch.concat(data_target, dim=1)


        for k, var in enumerate(variables_ordered):
            data_source[k] = self.var_normalizers[var].normalize(data_source[k])
        
        for k, var in enumerate(variables_ordered):
            data_target[k] = self.var_normalizers[var].normalize(data_target[k])

        data_source[masks] = 0
        data_source_zooms, mask_zooms = encode_zooms(data_source.float(), max_zoom, self.out_zooms, apply_diff=self.output_differences, mask=masks, binarize_mask=self.output_binary_mask)
        data_target_zooms = encode_zooms(data_target.float(), max_zoom, self.out_zooms, apply_diff=self.output_differences)[0]

        if self.mask_zooms:
            for zoom in self.mask_zooms:
                if zoom in data_source_zooms.keys():
                    data_source_zooms[zoom] = torch.zeros_like(data_target_zooms[zoom])
                    mask_zooms[zoom] = torch.ones_like(mask_zooms[zoom])
                    if self.output_binary_mask:
                        mask_zooms[zoom] = mask_zooms[zoom].bool()

        if self.zoom_patch_sample == -1:
            sample_dict = {
                }
        else:
            sample_dict = {
                'patch_index': patch_index,
                'zoom_patch_sample': self.zoom_patch_sample,
                }

        var_indices = torch.tensor(var_indices).unsqueeze(-1).repeat(1,data_source.shape[1])
        embed_data = {'VariableEmbedder': var_indices,
                      'DensityEmbedder': ({k: v.unsqueeze(dim=-1)for k, v in mask_zooms.items()}, var_indices),
                      'TimeEmbedder': time_source.float()}
        
        coords_input = torch.tensor([])
        coords_output = torch.tensor([])
        rel_dists_input = torch.tensor([])
        rel_dists_output = torch.tensor([])

        mask_zooms = {k: v.unsqueeze(dim=-1)for k, v in mask_zooms.items()}

        return data_source_zooms, data_target_zooms, coords_input, coords_output, sample_dict, mask_zooms, embed_data, rel_dists_input, rel_dists_output

    def __len__(self):
        return self.len_dataset