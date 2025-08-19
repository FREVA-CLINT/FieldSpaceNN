import copy
import json
from collections import defaultdict

import numpy as np
import torch
import xarray as xr
from omegaconf import ListConfig
from torch.utils.data import Dataset
from ..modules.grids.grid_utils import get_coords_as_tensor,get_grid_type_from_var,get_mapping_weights,encode_zooms,to_zoom, apply_zoom_diff, get_matching_time_patch

import warnings
warnings.filterwarnings("ignore", message=".*fails while guessing")

from . import normalizer as normalizers

def skewed_random_p(size, exponent=2, max_p=0.9):
    uniform_random = torch.rand(size)
    skewed_random =  max_p*(1 - uniform_random ** exponent)
    return skewed_random

def invert_dict(d):
    inverted_d = {}
    for key, value in d.items():
        inverted_d.setdefault(value, []).append(key)
    return inverted_d

#def create_mask(random_p, drop_mask, ):

class BaseDataset(Dataset):
    def __init__(self,
                 mapping_fcn=None,
                 norm_dict=None,
                 lazy_load=True,
                 mask_zooms=None,
                 p_dropout=0,
                 p_drop_groups=0,
                 n_drop_groups=-1,
                 random_p = False,
                 skewness_exp = 2,
                 n_sample_groups=-1,
                 deterministic=False,
                 output_binary_mask=False,
                 output_differences=True,
                 dropout_zooms=None,
                 reduce_zoom_to_batch=None,
                 reduce_time_to_batch=None
                 ):
        
        super(BaseDataset, self).__init__()
        
        self.norm_dict = norm_dict
        self.lazy_load = lazy_load
        self.random_p = random_p
        self.p_dropout = p_dropout
        self.skewness_exp = skewness_exp
        self.n_sample_groups = n_sample_groups
        self.deterministic = deterministic
        self.p_drop_groups = p_drop_groups
        self.n_drop_groups = n_drop_groups
        self.output_differences = output_differences
        self.output_binary_mask = output_binary_mask
        self.mask_zooms = mask_zooms
        self.reduce_zoom_to_batch = reduce_zoom_to_batch
        self.reduce_time_to_batch = reduce_time_to_batch

        self.variables = [v['variables'] for v in self.data_dict['variables'].values()]
        self.var_groups = [g for g in self.data_dict['variables'].keys()]
        self.var_tot_depths = [len(v['depths']) for v in self.data_dict['variables'].values()]

       # self.depths = [v['depths'] for v in self.data_dict['variables'].values()]

        if "files" in self.data_dict['source'].keys():
            all_files = self.data_dict['source']["files"]

        all_files = []
        for data in self.data_dict['source'].values():
            if isinstance(data['files'], list) or isinstance(data['files'], ListConfig):
                all_files += data['files']
            else:
                all_files.append(data['files'])
        
        self.zoom_time_steps_past = [v['n_past_ts'] for v in self.sampling_zooms.values()]
        self.zoom_time_steps_future = [v['n_future_ts'] for v in self.sampling_zooms.values()]
        self.zooms = [z for z in self.sampling_zooms.keys()]
        
        self.max_time_step_past = max(self.zoom_time_steps_past)
        self.max_time_step_future = max(self.zoom_time_steps_future)

        #TODO fix the variable ids if multi var training/inference
      #  self.variables_source_train = variables_source_train if variables_source_train is not None else self.variables_source
       # self.variables_target_train = variables_target_train if variables_target_train is not None else self.variables_target

    #    self.var_indices = dict(zip(self.variables_source_train, np.arange(len(self.variables_source_train))))

        if "timesteps" in self.data_dict.keys():
            self.sample_timesteps = []
            for t in self.data_dict["timesteps"]:
                if isinstance(t, int) or "-" not in t:
                    self.sample_timesteps.append(int(t))
                else:
                    start, end = map(int, t.split("-"))
                    self.sample_timesteps += list(range(start, end))
            self.sample_timesteps = self.sample_timesteps
        else:
            self.sample_timesteps = None

        self.time_steps_files = []
        for k, file in enumerate(self.data_dict['source'][self.zooms[0]]['files']):
            ds = xr.open_dataset(file)
            self.time_steps_files.append(len(ds.time))

        self.index_map = dict(zip(self.zooms, [[] for _ in self.zooms]))
        time_idx_global = self.max_time_step_past
        for file_idx, num_timesteps in enumerate(self.time_steps_files):
            if file_idx == len(self.time_steps_files) - 1:
                num_timesteps -= self.max_time_step_future

            start_idx = self.max_time_step_past if file_idx == 0 else 0

            for time_idx in range(start_idx, num_timesteps):
                if self.sample_timesteps is None or time_idx_global in self.sample_timesteps:
                    for zoom in self.zooms:
                        for region_idx_max in range(self.indices[max(self.zooms)].shape[0]):
                            if self.sampling_zooms[zoom]['zoom_patch_sample'] == -1:
                                region_idx_zoom = 0
                            else:
                                region_idx_zoom = region_idx_max//4**(self.sampling_zooms[max(self.zooms)]['zoom_patch_sample'] - self.sampling_zooms[zoom]['zoom_patch_sample'])

                            self.index_map[zoom].append((file_idx, time_idx, region_idx_zoom))
                
                time_idx_global += 1

        self.index_map = {z: np.array(idx_map) for z, idx_map in self.index_map.items()}

        all_variables = []
        group_ids = []
        for group_id, vars in enumerate(self.data_dict['variables'].values()):
            all_variables += vars['variables']
            group_ids.append(group_id)

        self.vars_group_ids = dict(zip(all_variables,group_ids))
        self.group_ids_vars = invert_dict(self.vars_group_ids)

        grid_types = [get_grid_type_from_var(ds, var) for var in all_variables]
        self.vars_grid_types = dict(zip(all_variables, grid_types))
        self.grid_types = np.unique(grid_types)

        self.grid_types_vars = invert_dict(self.vars_grid_types)

        for var, gtype in zip(all_variables, grid_types):
            self.grid_types_vars[gtype].append(var)

        unique_files = np.unique(np.array(all_files))
        
        if len(unique_files)==1:
            self.single_source = True
            coords = [get_coords_as_tensor(ds, grid_type=grid_type) for grid_type in self.grid_types]
            mapping_hr = dict(zip(self.grid_types, [mapping_fcn(coords_, max(self.zooms))[max(self.zooms)] for coords_ in coords]))
            self.mapping = {max(self.zooms): mapping_hr}
        else:
            self.single_source = False
            self.mapping = {}
            for zoom in self.zooms:
                mapping_grid_type = {}
                for grid_type in self.grid_types:
                    ds = xr.open_dataset(self.data_dict['source'][zoom]['files'][0])
                    coords = get_coords_as_tensor(ds, grid_type=grid_type)
                    mapping_grid_type[grid_type] = mapping_fcn(coords, zoom)[zoom]
                self.mapping[zoom] = mapping_grid_type

        with open(norm_dict) as json_file:
            norm_dict = json.load(json_file)

        self.var_normalizers = {}
        for var in all_variables:
            norm_class = norm_dict[var]['normalizer']['class']
            assert norm_class in normalizers.__dict__.keys(), f'normalizer class {norm_class} not defined'

            self.var_normalizers[var] = normalizers.__getattribute__(norm_class)(
                norm_dict[var]['stats'],
                norm_dict[var]['normalizer'])

        self.len_dataset = max([idx_map.shape[0] for idx_map in self.index_map.values()])
    
    def get_indices_from_patch_idx(self, patch_idx):
        raise NotImplementedError

    def get_files(self, file_path_source, file_path_target=None, drop_source=False):
      
        if self.lazy_load:
            ds_source = xr.open_dataset(file_path_source, decode_times=False)
        else:
            ds_source = xr.load_dataset(file_path_source, decode_times=False)

        if file_path_target is None:
            ds_target = copy.deepcopy(ds_source)

        elif file_path_target==file_path_source and not drop_source:
            ds_target = None
            
        else:
            if self.lazy_load:
                ds_target = xr.open_dataset(file_path_target, decode_times=False)
            else:
                ds_target = xr.load_dataset(file_path_target, decode_times=False)

        return ds_source, ds_target

    #def map_data(self):


    def get_data(self, ds, time_idx, patch_idx, variables_sample, mapping, mapping_zoom, zoom, group_ids_sample=None, drop_mask=None):
        # Fetch raw patch indices
        drop_mask_ = drop_mask.clone() if drop_mask is not None else None

        n_past_timesteps = self.sampling_zooms[zoom]['n_past_ts']
        n_future_timesteps = self.sampling_zooms[zoom]['n_future_ts']

        time_indices = np.arange(time_idx-n_past_timesteps, time_idx + n_future_timesteps + 1, 1)
        isel_dict = {"time": time_indices}
        patch_dim = [d for d in ds.dims if "cell" in d or "ncells" in d][0]

        nt = 1 + n_past_timesteps + n_future_timesteps

        data_g = []
        for grid_type, variables_grid_type in self.grid_types_vars.items():

            variables = [var for var in variables_sample if var in variables_grid_type]
            
            mapping = mapping[grid_type] 

            patch_indices = self.get_indices_from_patch_idx(zoom, patch_idx)

            mask = (1. * get_mapping_weights(mapping)[...,0]).view(1,1,-1,1)

            post_map = mapping_zoom > zoom
            if post_map:
                indices = mapping['indices'][...,[0]].reshape(-1,4**(mapping_zoom-zoom))
                isel_dict[patch_dim] = indices.view(-1)

            else:
                indices = mapping['indices'][...,[0]]
                mask = mask[:,:,patch_indices]
                isel_dict[patch_dim] = indices[patch_indices].view(-1)
                if drop_mask_ is not None:
                    drop_mask_ = drop_mask[:,:,patch_indices]                       
            
            ds = ds.isel(isel_dict)
            
            for i, variable in enumerate(variables):
                data = torch.tensor(ds[variable].values).view(nt, -1,1)
                data = self.var_normalizers[variable].normalize(data)
                data_g.append(data) 

        data_g = torch.stack(data_g, dim=0)
        
        _, counts = np.unique(group_ids_sample, return_counts=True)

        data_g = data_g.split(counts.tolist(), dim=0)
        data_g = torch.concat([data.transpose(0,-1).expand(-1,-1,-1, max(self.var_tot_depths)) for data in data_g],dim=0)

        if drop_mask_ is not None:
            mask = (1-1.*drop_mask_.unsqueeze(dim=-1)) * mask

        data_g, mask = to_zoom(data_g, mapping_zoom, zoom, mask=mask.expand_as(data_g), binarize_mask=self.output_binary_mask)

        if post_map:
            data_g = data_g[:,:,patch_indices]
            mask = mask[:,:,patch_indices]

        drop_mask = torch.logical_not(mask[...,[0]]) if mask.dtype==torch.bool else  1 - mask[...,[0]]

        data_time = torch.tensor(ds["time"].values).float()

        ds.close()

        return data_g, data_time, drop_mask

   #def get_masks_zooms(self, grid_type):

    def get_mask(self, ng, nt, n, p_dropout=0, p_drop_groups=0, p_drop_time_steps=0, n_drop_groups=-1):

        drop_groups = torch.rand(1) < p_drop_groups
        drop_timesteps = torch.rand(1) < p_drop_time_steps
        drop_mask = torch.zeros((ng, nt, n), dtype=bool)

        if self.random_p and drop_groups:
            p_dropout = skewed_random_p(ng, exponent=self.skewness_exp, max_p=p_dropout)
        elif self.random_p:
            p_dropout = skewed_random_p(1, exponent=self.skewness_exp, max_p=p_dropout)
        else:
            p_dropout = torch.tensor(p_dropout)

        if p_dropout > 0 and not drop_groups and not drop_timesteps:
            drop_mask_p = (torch.rand((nt, n)) < p_dropout).bool()
            drop_mask[:,drop_mask_p] = True

        elif p_dropout > 0 and drop_groups:
            drop_mask_p = (torch.rand((ng, nt, n)) < p_dropout.view(-1,1,1)).bool()
            drop_mask[drop_mask_p] = True

        elif p_dropout > 0 and drop_timesteps:
            drop_mask_p = (torch.rand(nt)<p_dropout).bool()
            drop_mask[:,drop_mask_p]=True

        if n_drop_groups!=-1 and n_drop_groups < ng:
            not_drop_vars = torch.randperm(ng)[:(ng-n_drop_groups)]
            drop_mask[not_drop_vars] = (drop_mask[not_drop_vars]*0).bool()

        return drop_mask
  
    

    def __getitem__(self, index):
        
        if self.n_sample_groups != -1 and self.n_sample_groups != len(self.variables):
            group_indices = np.random.randint(0,len(self.variables), (self.n_sample_groups))
        else:
            group_indices = np.arange(len(self.variables))
        
        variables_sample = []
        group_ids_sample = []

        for k in group_indices:
            variables_sample += self.variables[k]
            group_ids_sample += [k]*len(self.variables[k])


        if self.single_source and self.p_dropout > 0:
            nt = 1 + self.max_time_step_future + self.max_time_step_past
            drop_mask_input = self.get_mask(len(group_indices), 
                                            nt, 
                                            self.indices[max(self.zooms)].size,
                                            self.p_dropout,
                                            self.p_drop_groups,
                                            self.n_drop_groups)
        else:
            drop_mask_input = None
            if self.p_dropout > 0:
                UserWarning('Multi-source input does not support global dropout')

        target_zooms = {}    
        source_zooms = {}
        time_zooms = {}
        mask_mapping_zooms = {}
        for zoom in self.zooms:
            file_index, time_index, patch_index = self.index_map[zoom][index]

            if self.single_source:
                source_file = self.data_dict['source'][max(self.zooms)]['files'][int(file_index)]
                target_file = self.data_dict['target'][max(self.zooms)]['files'][int(file_index)]
                mapping_zoom = max(self.zooms)
                
            else:
                source_file = self.data_dict['source'][zoom]['files'][int(file_index)]
                target_file = self.data_dict['target'][zoom]['files'][int(file_index)]
                mapping_zoom = zoom

            ds_source, ds_target = self.get_files(source_file, file_path_target=target_file, drop_source=self.p_dropout>0)

            if drop_mask_input is not None:
                ts_start = self.max_time_step_past - self.sampling_zooms[zoom]['n_past_ts']
                ts_end = self.max_time_step_future - self.sampling_zooms[zoom]['n_future_ts']
                drop_mask_zoom = drop_mask_input[:,ts_start:(drop_mask_input.shape[1]-ts_end)]
            else:
                drop_mask_zoom = None

            data_source, data_time_zoom, drop_mask_zoom = self.get_data(ds_source, 
                                                                 time_index, 
                                                                 patch_index, 
                                                                 variables_sample, 
                                                                 self.mapping[max(self.zooms)], 
                                                                 mapping_zoom, 
                                                                 zoom, 
                                                                 group_ids_sample=np.array(group_ids_sample), 
                                                                 drop_mask=drop_mask_zoom)
            
            if ds_target is not None:
                data_target, _, _ = self.get_data(ds_target, 
                                                time_index, 
                                                patch_index, 
                                                variables_sample, 
                                                self.mapping[max(self.zooms)], 
                                                mapping_zoom, 
                                                zoom, 
                                                group_ids_sample=np.array(group_ids_sample))
            else:
                data_target = data_source.clone()

            source_zooms[zoom] = data_source
            target_zooms[zoom] = data_target
            time_zooms[zoom] = data_time_zoom
            mask_mapping_zooms[zoom] = drop_mask_zoom

        sample_configs = self.sampling_zooms
        for zoom in self.sampling_zooms.keys():
            sample_configs[zoom]['patch_index'] = int(self.index_map[zoom][index][-1])
        
        patch_index_zooms = {}
        for zoom in self.sampling_zooms.keys():
            patch_index_zooms[zoom] = torch.tensor(self.index_map[zoom][index][-1])


        data_source = apply_zoom_diff(source_zooms, self.sampling_zooms)
        data_target = apply_zoom_diff(target_zooms, self.sampling_zooms)

        for zoom in data_source.keys():
            if mask_mapping_zooms[zoom].dtype == torch.float:
                mask_zoom = mask_mapping_zooms[zoom] == 1
            else:
                mask_zoom = mask_mapping_zooms[zoom]
            data_source[zoom][mask_zoom.expand_as(data_source[zoom])] = 0

        mask_zooms = mask_mapping_zooms

        embed_data = {'GroupEmbedder': torch.tensor(group_indices),
                      'DensityEmbedder': ({k: v for k, v in mask_zooms.items()}, torch.tensor(group_indices)),
                      'TimeEmbedder': data_time_zoom}
        

        sample_configs = torch.tensor([])

        return source_zooms, target_zooms, patch_index_zooms, mask_zooms, embed_data

    def __len__(self):
        return self.len_dataset