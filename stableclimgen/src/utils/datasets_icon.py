import copy

from typing import Any

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset, Sampler
from .grid_utils_icon import get_coords_as_tensor, get_nh_variable_mapping_icon


class InfiniteSampler(Sampler):
    """
    A PyTorch Sampler that provides an endless stream of random indices, allowing for 
    continuous sampling over a finite dataset without resetting, especially useful for 
    training on large datasets in an "infinite" manner.

    Parameters:
    ----------
    num_samples : int
        The number of samples in the dataset to sample from.
    data_source : Dataset, optional
        The dataset from which to sample. This parameter is required by the base 
        `Sampler` class but is not used directly in `InfiniteSampler` (default is None).

    Methods:
    -------
    __iter__()
        Returns an iterator that yields indices indefinitely, looping over shuffled indices.

    __len__()
        Returns an arbitrarily large length, simulating an infinite sampling size.

    loop()
        Generator function that yields indices from a random permutation of `num_samples`.
        Reshuffles and restarts when all indices have been yielded.

    Usage:
    ------
    Used with PyTorch DataLoader for training tasks that require continuous sampling.
    """
    def __init__(self, num_samples, data_source=None):
        super().__init__(data_source)
        self.num_samples = num_samples

    def __iter__(self):
        return iter(self.loop())

    def __len__(self):
        return 2 ** 31

    def loop(self):
        i = 0
        n_samples = self.num_samples 
        order = np.random.permutation(n_samples) 
        while True:
            yield order[i]
            i += 1
            if i >= n_samples:
                order = np.random.permutation(n_samples) 
                i = 0

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


class NetCDFLoader_lazy(Dataset):
    def __init__(self, data_dict,
                 normalizer,
                 grid_processing,
                 coarsen_sample_level,
                 search_radius=2,
                 nh_input=1,
                 index_range_source=None,
                 index_offset_target=0,
                 sample_for_norm=-1,
                 lazy_load=False,
                 random_time_idx=True,
                 p_dropout=0,
                 p_average=0,
                 p_average_dropout=0,
                 max_average_lvl=0,
                 drop_vars=False,
                 n_sample_vars=-1):
        
        super(NetCDFLoader_lazy, self).__init__()
        
        self.coarsen_sample_level = coarsen_sample_level
        self.normalizer = normalizer
        self.index_range_source=index_range_source
        self.index_offset_target = index_offset_target
        self.sample_for_norm = sample_for_norm
        self.lazy_load=lazy_load
        self.random_time_idx = random_time_idx
        self.p_dropout = p_dropout
        self.p_average = p_average
        self.p_average_dropout = p_average_dropout
        self.max_average_lvl = max_average_lvl
        self.drop_vars = drop_vars
        self.n_sample_vars = n_sample_vars

        self.variables_source = data_dict["source"]["variables"]
        self.variables_target = data_dict["target"]["variables"]
        self.files_source = np.loadtxt(data_dict["source"]["files"],dtype='str')
        self.files_target = np.loadtxt(data_dict["target"]["files"],dtype='str')
        grid_input = data_dict["source"]["grid"]
        grid_output = data_dict["target"]["grid"]

        coords_processing = get_coords_as_tensor(xr.open_dataset(grid_processing), lon='clon', lat='clat', target='numpy')

        
        if  grid_input != grid_processing:
            input_mapping, input_in_range = get_nh_variable_mapping_icon(grid_processing, 
                                                                        ['cell'], 
                                                                        grid_input, 
                                                                        ['cell'], 
                                                                        search_radius=search_radius, 
                                                                        max_nh=nh_input,
                                                                        lowest_level=0,
                                                                        coords_icon=coords_processing,
                                                                        scale_input=1.,
                                                                        periodic_fov= None)
                            
            input_mapping = input_mapping['cell']['cell']
            input_in_range = input_in_range['cell']['cell']
            input_coordinates = get_coords_as_tensor(grid_input, lon='clon', lat='clat', target='numpy')
        else:
            input_mapping = np.arange(coords_processing.shape[1])[:,np.newaxis]
            input_in_range = np.ones_like(input_mapping, dtype=bool)[:,np.newaxis]
            input_coordinates = None
            
        if grid_output != grid_processing:
            output_mapping, output_in_range = get_nh_variable_mapping_icon(
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
            output_in_range = output_in_range['cell']['cell']
            output_coordinates = get_coords_as_tensor(grid_output, lon='clon', lat='clat', target='numpy')
        else:
            output_mapping = np.arange(coords_processing.shape[1])[:,np.newaxis]
            output_in_range = np.ones_like(output_mapping, dtype=bool)[:,np.newaxis]
            output_coordinates = None
            
        
        self.input_mapping = input_mapping
        self.input_in_range = input_in_range
        self.output_mapping = output_mapping
        self.output_in_range = output_in_range
        self.input_coordinates = input_coordinates
        self.output_coordinates = output_coordinates
        
        global_indices = np.arange(coords_processing.shape[1])

        self.global_cells = global_indices.reshape(-1,4**coarsen_sample_level)
        self.global_cells_input = self.global_cells[:,0]
            
      
        ds_source = xr.open_dataset(self.files_source[0])
        self.len_dataset = len(ds_source)*self.global_cells.shape[1] #ds_source[self.variables_source[0]].shape[0]

        self.normalizer = normalizer
    
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


    def get_data(self, ds, ts, global_indices, variables, global_level_start, index_mapping_dict=None):
        
        if index_mapping_dict is not None:
            indices = index_mapping_dict[global_indices // 4**global_level_start] 
        else:
            indices = global_indices.view(-1,1)

        n, nh = indices.shape
        indices = indices.reshape(-1)

        # tbd: spatial dim as input!
        if 'ncells' in dict(ds.sizes).keys():
            ds = ds.isel(ncells=indices, time=ts)
        else:
            ds = ds.isel(cell=indices, time=ts)

        data_g = []
        for variable in variables:
            data = torch.tensor(ds[variable].values)
            data = data[0] if data.dim() > 1  else data
            data_g.append(data)

        data_g = torch.stack(data_g, dim=-1)
        data_g = data_g.view(n, nh, len(variables), 1)

        ds.close()

        return data_g

    def __getitem__(self, index):
        
        if len(self.files_source)>0:
            source_index = torch.randint(0, len(self.files_source), (1,1))
            source_file = self.files_source[source_index]   
        
        target_file = self.files_target[source_index]

        
        ds_source, ds_target = self.get_files(source_file, file_path_target=target_file)

        if self.random_time_idx:
            index = int(torch.randint(0, len(ds_source.time.values), (1,1)))
            if self.index_range_source is not None:
                if (index < self.index_range_source[0]) or (index > self.index_range_source[1]):
                    index = int(torch.randint(self.index_range_source[0], self.index_range_source[1]+1, (1,1)))


        global_cells_input = self.global_cells_input
        input_mapping = self.input_mapping
        global_cells = self.global_cells
        input_in_range = self.input_in_range
        output_mapping = self.output_mapping
        output_in_range = self.output_in_range

    
        sample_index = torch.randint(global_cells_input.shape[0],(1,))[0]

        if self.n_sample_vars !=-1 and self.n_sample_vars != len(self.variables_source):
            sample_vars = torch.randperm(len(self.variables_source))[:self.n_sample_vars]
        else:
            sample_vars = torch.arange(len(self.variables_source))

        self.variables_source = np.array(self.variables_source)[sample_vars]
        self.variables_target = np.array(self.variables_target)[sample_vars]

        data_source = self.get_data(ds_source, index, global_cells[sample_index] , self.variables_source, 0, input_mapping)

        if ds_target is not None:
            data_target = self.get_data(ds_target, index, global_cells[sample_index] , self.variables_target, 0, output_mapping)
        else:
            data_target = data_source

        indices = {'global_cell': torch.tensor(global_cells[sample_index]),
                    'local_cell': torch.tensor(global_cells[sample_index]),
                    'sample': sample_index,
                    'sample_level': self.coarsen_sample_level,
                    'variables': sample_vars}

        if self.input_coordinates is not None:
            coords_input = torch.tensor(self.input_coordinates[input_mapping[global_cells[sample_index]]])
        else:
            coords_input = torch.tensor([])

        if self.output_coordinates is not None:
            coords_output = torch.tensor(self.output_coordinates[output_mapping[global_cells[sample_index]]])
        else:
            coords_output = torch.tensor([])

        n, nh, nv, f = data_source.shape
        if self.p_average > 0 and torch.rand(1)<self.p_average:
            avg_level = int(torch.randint(1,self.max_average_lvl+1,(1,)))
            data_source_resh = data_source.view(-1,4**avg_level,nh,f)
            data_source_resh = data_source_resh.mean(dim=[1,2], keepdim=True)
            data_source_resh = data_source_resh.repeat_interleave(4**avg_level, dim=1)
            data_source_resh = data_source_resh.repeat_interleave(nh, dim=2)
            data_source = data_source_resh.view(n, nh, nv, f)

            drop_mask = torch.zeros((n,nh,nv), dtype=bool)

            if self.p_average_dropout >0:
                if self.drop_vars:
                    drop_mask_p = (torch.rand((n//4**avg_level,nv))<self.p_average_dropout).bool()
                else:
                    drop_mask_p = (torch.rand((n//4**avg_level))<self.p_average_dropout).bool()

                drop_mask = drop_mask.view(-1,4**avg_level, nh, nv).transpose(-1,1)
                drop_mask[drop_mask_p]=True
                drop_mask = drop_mask.transpose(-1,1).view(-1,nh,nv)
        else:
            drop_mask = torch.zeros((n,nh,nv), dtype=bool)

            if self.p_dropout > 0 and not self.drop_vars:
                drop_mask_p = (torch.rand((n,nh))<self.p_dropout).bool()
                drop_mask[drop_mask_p]=True

            elif self.drop_vars:
                drop_mask_p = (torch.rand((n,nh,nv))<self.p_dropout).bool()
                drop_mask[drop_mask_p]=True

    
        if self.normalizer is not None:
            for k, var in enumerate(self.variables_source):
                data_source[:,:,k,:] = self.normalizer.normalize(data_source[:,:,k,:], var)
            
            for k, var in enumerate(self.variables_target):
                data_target[:,:,k,:] = self.normalizer.normalize(data_target[:,:,k,:], var)

        data_source[drop_mask] = 0

        ds_target = ds_source = output_mapping = input_mapping = global_cells = global_cells = []

        return data_source.float(), data_target.float(), indices, drop_mask, coords_input.float(), coords_output.float()

    def __len__(self):
        return self.len_dataset