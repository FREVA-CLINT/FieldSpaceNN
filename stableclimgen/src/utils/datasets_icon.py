import copy

import json
from typing import Any

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset, Sampler
from .grid_utils_icon import get_coords_as_tensor, get_nh_variable_mapping_icon, get_grid_type_from_var
from . import normalizer as normalizers


def skewed_random_p(size, exponent=2, max_p=0.9):
    uniform_random = torch.rand(size)
    skewed_random =  max_p*(1 - uniform_random ** exponent)
    return skewed_random


def get_moments(data, type, level=0.9):

    if type == 'quantile':
        moments = (np.quantile((data), ((1-level), level)).astype(float))

    elif type == 'quantile_abs':
        q = np.quantile(np.abs(data), level).astype(float)
        moments = (q,q)

    elif type == 'None':
        moments = (1., 1.)

    elif type == 'min_max':
        moments = (data.min().astype(float), data.max().astype(float))
    else:
        moments = (data.mean().astype(float), data.std().astype(float))
    
    return tuple(moments)



def get_stats(files, variable, norm_dict, n_sample=None):
    if (n_sample is not None) and (n_sample < len(files)):
        file_indices = np.random.randint(0,len(files), (n_sample,))
    else:
        file_indices = np.arange(len(files))
    
    if variable == "uv":
        variables = ["u", "v"]
    else:
        variables = [variable]
    
    data = []
    for file_index in np.array(files)[file_indices]:
        ds = xr.load_dataset(file_index)
        for variable in variables:
            d = ds[variable].values
            if len(d.shape)==3:
                d = d[0,0]
            elif len(d.shape)==2: 
                d = d[0]
            data.append(d.flatten())

    return get_moments(np.concatenate(data), norm_dict["type"], level=norm_dict["level"])

def invert_dict(dict):
    dict_out = {}
    unique_values = np.unique(np.array(list(dict.values())))

    for uni_value in unique_values:
        dict_out[uni_value] = [key for key,value in dict.items() if value==uni_value]
    return dict_out

class NetCDFLoader_lazy(Dataset):
    def __init__(self, data_dict,
                 norm_dict,
                 grid_processing: str,
                 coarsen_sample_level,
                 search_radius=2,
                 nh_input=1,
                 index_range_source=None,
                 index_offset_target=0,
                 sample_for_norm=-1,
                 lazy_load=False,
                 random_time_idx=True,
                 p_dropout=0,
                 random_p = True,
                 skewness_exp = 2,
                 p_average=0,
                 p_average_dropout=0,
                 max_average_lvl=0,
                 p_drop_vars=0,
                 n_drop_vars=-1,
                 n_sample_vars=-1,
                 pert_coordinates=0,
                 fixed_sample_ids=None,
                 fixed_seed = False,
                 variables_source=None,
                 variables_target=None):
        
        super(NetCDFLoader_lazy, self).__init__()
        
        self.fixed_sample_ids = fixed_sample_ids
        self.fixed_seed = fixed_seed
        self.coarsen_sample_level = coarsen_sample_level
        self.norm_dict = norm_dict
        self.index_range_source=index_range_source
        self.index_offset_target = index_offset_target
        self.sample_for_norm = sample_for_norm
        self.lazy_load=lazy_load
        self.random_time_idx = random_time_idx
        self.p_dropout = p_dropout
        self.random_p = random_p
        self.skewness_exp = skewness_exp
        self.p_average = p_average
        self.p_average_dropout = p_average_dropout
        self.max_average_lvl = max_average_lvl
        self.p_drop_vars = p_drop_vars
        self.n_sample_vars = n_sample_vars
        self.pert_coordinates = pert_coordinates
        self.n_drop_vars = n_drop_vars

        self.variables_source = data_dict["source"]["variables"]
        self.variables_target = data_dict["target"]["variables"]

        self.variables_source_train = variables_source if variables_source is not None else self.variables_source
        self.variables_target_train = variables_target if variables_target is not None else self.variables_target

        if "timesteps" in data_dict.keys():
            self.sample_timesteps = []
            for t in data_dict["timesteps"]:
                if type(t) == int or "-" not in t:
                    self.sample_timesteps.append(int(t))
                else:
                    self.sample_timesteps += list(range(int(t.split("-")[0]), int(t.split("-")[1])))
        else:
            self.sample_timesteps = None      

        with open(norm_dict) as json_file:
            norm_dict = json.load(json_file)

        self.var_normalizers = {}
        for var in self.variables_source:
            norm_class = norm_dict[var]['normalizer']['class']
            assert norm_class in normalizers.__dict__.keys(), f'normalizer class {norm_class} not defined'

            self.var_normalizers[var] = normalizers.__getattribute__(norm_class)(
                norm_dict[var]['stats'],
                norm_dict[var]['normalizer'])


        self.files_source = np.loadtxt(data_dict["source"]["files"],dtype='str')
        self.files_target = np.loadtxt(data_dict["target"]["files"],dtype='str')

        """
        permutation_indices =  np.random.permutation(np.arange(len(self.files_source)))
        self.files_source = self.files_source[permutation_indices]
        self.files_target = self.files_target[permutation_indices]
        """

        grid_input = data_dict["source"]["grid"]
        grid_output = data_dict["target"]["grid"]

        coords_processing = get_coords_as_tensor(xr.open_dataset(grid_processing), lon='clon', lat='clat', target='numpy')
        
        self.grid_types_vars_input = [get_grid_type_from_var(xr.open_dataset(self.files_source[0]), variable) for variable in self.variables_source]
        grid_types_vars_unique = np.unique(np.array(self.grid_types_vars_input))
        
        self.vars_grid_types_input = invert_dict(dict(zip(self.variables_source, self.grid_types_vars_input)))

        if  grid_input != grid_processing:
            input_mapping, input_in_range, positions = get_nh_variable_mapping_icon(grid_processing, 
                                                                        ['cell'], 
                                                                        grid_input, 
                                                                        grid_types_vars_unique, 
                                                                        search_radius=search_radius, 
                                                                        max_nh=nh_input,
                                                                        lowest_level=0,
                                                                        coords_icon=coords_processing,
                                                                        scale_input=1.,
                                                                        periodic_fov= None)
                            
            input_mapping = input_mapping['cell']
           # input_in_range = input_in_range['cell']
            input_coordinates = {}
            for grid_type in grid_types_vars_unique:
                input_coordinates[grid_type] = get_coords_as_tensor(xr.open_dataset(grid_input), grid_type=grid_type, target='numpy')
            
            input_positions = positions['cell']
        else:
            input_mapping = {'cell': np.arange(coords_processing.shape[0])[:,np.newaxis]}
            #input_in_range = {'cell': np.ones_like(input_mapping['cell'], dtype=bool)}
            input_coordinates = None
            input_positions = None
            
        if grid_output != grid_processing:
            output_mapping, output_in_range, positions = get_nh_variable_mapping_icon(
                                                        grid_processing, ['cell'], 
                                                        grid_input, 
                                                        ['cell'], 
                                                        search_radius=search_radius, 
                                                        max_nh=nh_input,
                                                        lowest_level=0,
                                                        coords_icon=coords_processing,
                                                        scale_input=1.,
                                                        periodic_fov= None)
            
            output_mapping = output_mapping['cell']['cell']
            positions = positions['cell']['cell']
            output_coordinates = get_coords_as_tensor(xr.open_dataset(grid_output), lon='clon', lat='clat', target='numpy')
        else:
            output_mapping = {'cell': np.arange(coords_processing.shape[0])[:,np.newaxis]}
            output_coordinates = None
            
        self.input_positions = input_positions
        self.input_mapping = input_mapping
        self.output_mapping = output_mapping
        self.input_coordinates = input_coordinates
        self.output_coordinates = output_coordinates
        
        global_indices = np.arange(coords_processing.shape[0])

        if coarsen_sample_level == -1:
            self.global_cells = global_indices.reshape(1, -1)
            self.global_cells_input = np.array([1]).reshape(-1,1)
        else:
            self.global_cells = global_indices.reshape(-1, 4**coarsen_sample_level)
            self.global_cells_input = self.global_cells[:,0]
            

        self.n_files = len(self.files_source)
        self.n_regions = self.global_cells.shape[0]

        self.global_indices = []
        idx = 0
        for k, file in enumerate(self.files_source):
            ds = xr.open_dataset(file)
            if self.sample_timesteps:
                idx_add = len(self.sample_timesteps) * self.n_regions
            else:
                idx_add = len(ds.time) * self.n_regions
            self.global_indices.append(idx + idx_add)
            idx += idx_add
        
        self.var_sizes_source = {}
        for var in self.variables_source:
            self.var_sizes_source[var] = list(ds[var].sizes)

        self.var_sizes_target = {}
        ds = xr.open_dataset(self.files_target[0])
        for var in self.variables_source:
            self.var_sizes_target[var] = list(ds[var].sizes)

        self.global_indices = np.array(self.global_indices)
        self.len_dataset = self.global_indices[-1]


    def get_files(self, file_path_source, file_path_target=None):
      

        if self.lazy_load:
            ds_source = xr.open_dataset(file_path_source)
        else:
            ds_source = xr.load_dataset(file_path_source)

        if file_path_target is None:
            ds_target = copy.deepcopy(ds_source)

        elif file_path_target==file_path_source:
            ds_target = ds_source
            
        else:
            if self.lazy_load:
                ds_target = xr.open_dataset(file_path_target)
            else:
                ds_target = xr.load_dataset(file_path_target)       

        return ds_source, ds_target


    def get_data(self, ds, ts, global_indices, variables, global_level_start, grid_type, index_mapping_dict=None):
        

        if index_mapping_dict is not None:
            indices = index_mapping_dict[global_indices // 4**global_level_start] 
        else:
            indices = global_indices.view(-1,1)

        n, nh = indices.shape
        indices = indices.reshape(-1)

        data_g = []

        if grid_type == 'lonlat':
            for variable in variables:
                data = torch.tensor(ds[variable][ts].values)

                data = data[0] if data.dim() > 2  else data
                data = data.reshape(-1)
                data_g.append(data[indices].reshape(-1, nh))

        else:
            isel_dict = {}
            if 'ncells' in dict(ds.sizes).keys():
                isel_dict['ncells'] = indices
            elif 'cell' in dict(ds.sizes).keys():
                isel_dict['cell'] = indices

            if 'time' in dict(ds.sizes).keys():
                isel_dict['time'] = ts

            if 'plev' in dict(ds.sizes).keys():
                isel_dict['plev'] = [0]
            
            ds = ds.isel(isel_dict)
    
            for variable in variables:

                data = torch.tensor(ds[variable].values)
                data = data[0] if data.dim() > 1  else data
                data_g.append(data)

        data_g = torch.stack(data_g, dim=-1)
        data_g = data_g.view(1, n, nh, len(variables), 1)

        ds.close()

        return data_g
    

    def __getitem__(self, index):
        if self.fixed_seed:
            torch.manual_seed(42)
            np.random.seed(42)
        if self.fixed_sample_ids is not None:
            index = self.fixed_sample_ids[np.random.randint(0,len(self.fixed_sample_ids))]

        diff = (self.global_indices - 1) - index
        diff[diff<0]=1e10
        file_idx = np.abs(diff).argmin()

        start_idx = self.global_indices[file_idx-1] if file_idx >0 else 0
        index_in_file = (index - start_idx)
        time_point_idx = index_in_file // self.n_regions
        region_idx = index_in_file - time_point_idx*self.n_regions

        source_file = self.files_source[file_idx]   
        target_file = self.files_target[file_idx]

        ds_source, ds_target = self.get_files(source_file, file_path_target=target_file)

        global_cells_input = self.global_cells_input
        input_mapping = self.input_mapping
        global_cells = self.global_cells
    #    input_in_range = self.input_in_range
        output_mapping = self.output_mapping
     #   output_in_range = self.output_in_range

        if self.n_sample_vars !=-1 and self.n_sample_vars != len(self.variables_source):
            sample_vars = torch.randperm(len(self.variables_source))[:self.n_sample_vars]
        else:
            sample_vars = torch.arange(len(self.variables_source))

        variables_source = np.array(self.variables_source)[sample_vars]
        variables_target = np.array(self.variables_target)[sample_vars]

        if len(sample_vars)==1:
            variables_source = [variables_source]
            variables_target = [variables_target]

        data_source_grids = {}
        for grid_type, variables in self.vars_grid_types_input.items():
            variables = [var for var in variables_source if var in variables]
            data_source = self.get_data(ds_source, time_point_idx, global_cells[region_idx], variables, 0, grid_type, input_mapping[grid_type])
            data_source_grids[grid_type] = data_source 

        data_source = torch.concat(list(data_source_grids.values()), dim=-2)

        if ds_target is not None:
            data_target = self.get_data(ds_target, time_point_idx, global_cells[region_idx] , variables_target, 0, 'cell',output_mapping['cell'])
        else:
            data_target = data_source

        ds_source.close()
        ds_target.close()   

        if self.input_coordinates is not None:
            grid_type = self.grid_types_vars_input[0]
            coords_input = torch.tensor(self.input_coordinates[grid_type])[self.input_mapping[grid_type][global_cells[region_idx]]].unsqueeze(dim=0)
        else:
            coords_input = torch.tensor([])

        if self.input_positions is not None:
            grid_type = self.grid_types_vars_input[0]
            dists_input = self.input_positions[grid_type][0][global_cells[region_idx]].unsqueeze(dim=0)
        else:
            dists_input = torch.tensor([])

        if self.output_coordinates is not None:
            coords_output = torch.tensor(self.output_coordinates[self.output_mapping[global_cells[region_idx]]])
        else:
            coords_output = torch.tensor([])

        nt, n, nh, nv, f = data_source.shape
        
        drop_vars = torch.rand(1)<self.p_drop_vars
        drop_mask = torch.zeros((nt, n,nh,nv), dtype=bool)

        if self.p_average > 0 and torch.rand(1)<self.p_average:
            avg_level = int(torch.randint(1,self.max_average_lvl+1,(1,)))
            data_source_resh = data_source.view(nt, -1,4**avg_level,nh,f)
            data_source_resh = data_source_resh.mean(dim=[2,3], keepdim=True)
            data_source_resh = data_source_resh.repeat_interleave(4**avg_level, dim=2)
            data_source_resh = data_source_resh.repeat_interleave(nh, dim=3)
            data_source = data_source_resh.view(nt, n, nh, nv, f)


            if self.p_average_dropout >0:
                if drop_vars:
                    drop_mask_p = (torch.rand((n//4**avg_level,nv))<self.p_average_dropout).bool()
                else:
                    drop_mask_p = (torch.rand((n//4**avg_level))<self.p_average_dropout).bool()

                drop_mask = drop_mask.view(nt, -1,4**avg_level, nh, nv).transpose(-1,2)
                drop_mask[drop_mask_p]=True
                drop_mask = drop_mask.transpose(-1,2).view(nt, -1,nh,nv)
        else:

            if self.random_p and drop_vars:
                p_dropout = skewed_random_p(nv, exponent=self.skewness_exp, max_p=self.p_dropout)
            elif self.random_p:
                p_dropout = skewed_random_p(1, exponent=self.skewness_exp, max_p=self.p_dropout)
            else:
                p_dropout = torch.tensor(self.p_dropout)

            if self.p_dropout > 0 and not drop_vars:
                drop_mask_p = (torch.rand((nt,n,nh))<p_dropout).bool()
                drop_mask[drop_mask_p]=True

            elif self.p_dropout > 0 and drop_vars:
                drop_mask_p = (torch.rand((nt,n,nh,nv))<p_dropout).bool()
                drop_mask[drop_mask_p]=True

        if self.n_drop_vars!=-1 and self.n_drop_vars < nv:
            not_drop_vars = torch.randperm(nv)[:(nv-self.n_drop_vars)]
            drop_mask[:,:,:,not_drop_vars] = (drop_mask[:,:,:,not_drop_vars]*0).bool()

        for k, var in enumerate(variables_source):
            data_source[:,:,:,k,:] = self.var_normalizers[var].normalize(data_source[:,:,:,k,:])
        
        for k, var in enumerate(variables_target):
            data_target[:,:,:,k,:] = self.var_normalizers[var].normalize(data_target[:,:,:,k,:])

        if hasattr(self,'input_in_range'):
            input_in_range = input_in_range[grid_type][global_cells[region_idx]]
            drop_mask[input_in_range==False] = True

        data_source[drop_mask] = 0

        ds_target = ds_source = output_mapping = input_mapping = global_cells = global_cells = []

        if self.coarsen_sample_level==-1:
            indices_sample= torch.tensor([])
        else:
            indices_sample = {'sample': region_idx,
                'sample_level': self.coarsen_sample_level}
        
        var_indices = torch.tensor([np.where(np.array(self.variables_source_train)==var)[0][0] for var in variables_source])
        embed_data = {'VariableEmbedder': var_indices.unsqueeze(dim=0)}

        return data_source.float(), data_target.float(), coords_input.float(), coords_output.float(), indices_sample, drop_mask, embed_data, dists_input

    def __len__(self):
        return self.len_dataset