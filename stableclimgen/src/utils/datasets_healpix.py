import copy
import json

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset
import healpy as hp

from . import normalizer as normalizers
from .datasets_icon import skewed_random_p
from .grid_utils_healpix import healpix_pixel_lonlat_torch, get_mapping_to_healpix_grid, get_coords_as_tensor


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


class HealPixLoader(Dataset):
    def __init__(self, data_dict,
                 coarsen_sample_level,
                 processing_nside,
                 norm_dict=None,
                 in_grid = None,
                 out_grid = None,
                 in_nside = None,
                 out_nside = None,
                 bottleneck_nside = None,
                 search_radius=2,
                 search_level_start=None,
                 search_level_stop=0,
                 nh_input=1,
                 index_offset_target=0,
                 sample_for_norm=-1,
                 lazy_load=False,
                 random_time_idx=True,
                 p_dropout=0,
                 p_average=0,
                 p_average_dropout=0,
                 max_average_lvl=0,
                 n_sample_vars=-1,
                 n_sample_timesteps=-1,
                 random_p = False,
                 skewness_exp = 2,
                 deterministic=False,
                 variables_source=None,
                 variables_target=None,
                 max_in_grid_distance=None,
                 one_to_one=False,
                 p_drop_vars=0,
                 n_drop_vars=-1,
                 p_drop_timesteps=-1
                 ):
        
        super(HealPixLoader, self).__init__()
        
        self.coarsen_sample_level = coarsen_sample_level
        self.norm_dict = norm_dict
        self.index_offset_target = index_offset_target
        self.sample_for_norm = sample_for_norm
        self.lazy_load=lazy_load
        self.random_time_idx = random_time_idx
        self.random_p = random_p
        self.p_dropout = p_dropout
        self.p_average = p_average
        self.p_average_dropout = p_average_dropout
        self.max_average_lvl = max_average_lvl
        self.skewness_exp = skewness_exp
        self.n_sample_vars = n_sample_vars
        self.n_sample_timesteps = n_sample_timesteps if n_sample_timesteps != -1 else 1
        self.deterministic = deterministic
        self.n_drop_vars = n_drop_vars
        self.p_drop_vars = p_drop_vars
        self.p_drop_timesteps = p_drop_timesteps
        self.bottleneck_nside = bottleneck_nside

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

        if self.norm_dict:
            with open(norm_dict) as json_file:
                norm_dict = json.load(json_file)

            self.var_normalizers = {}
            for var in self.variables_source:
                norm_class = norm_dict[var]['normalizer']['class']
                assert norm_class in normalizers.__dict__.keys(), f'normalizer class {norm_class} not defined'

                self.var_normalizers[var] = normalizers.__getattribute__(norm_class)(
                    norm_dict[var]['stats'],
                    norm_dict[var]['normalizer'])


        self.files_source = data_dict["source"]["files"]
        self.files_target = data_dict["target"]["files"]

        self.steady_mask = None

        coords_processing = healpix_pixel_lonlat_torch(processing_nside)

        if processing_nside == in_nside:
            input_mapping = np.arange(coords_processing.shape[0])[:, np.newaxis]
            input_coordinates = None
            input_positions = None
        else:
            if in_grid is not None:
                input_coordinates = get_coords_as_tensor(xr.open_dataset(in_grid),
                                                         grid_type='regular')  # change for other grids
            else:
                input_coordinates = healpix_pixel_lonlat_torch(in_nside)

            mapping = get_mapping_to_healpix_grid(coords_processing,
                                                  input_coordinates,
                                                  search_radius=search_radius,
                                                  search_level_start=search_level_start,
                                                  max_nh=nh_input,
                                                  lowest_level=search_level_stop,
                                                  periodic_fov=None)
            if one_to_one or max_in_grid_distance:
                # Compute pairwise distances (using broadcasting)
                diff = coords_processing.unsqueeze(1) - input_coordinates.unsqueeze(0)   # Shape: (grid_size, num_in, 2)

                # Compute Euclidean distance (or Haversine if needed)
                distances = torch.norm(diff, dim=-1)  # Shape: (grid_size, num_in)

                if one_to_one:
                    closest_out_indices = torch.argmin(distances, dim=0)  # One closest out_coord per in_coord
                    self.steady_mask = torch.ones(coords_processing.shape[0], dtype=bool)
                    self.steady_mask[closest_out_indices] = False
                else:
                    # Check if any input_coordinates are within d
                    self.steady_mask = ~(distances <= max_in_grid_distance).any(dim=1)
                print("Dropout percentage: {:.2f}".format((self.steady_mask == True).float().mean().item() * 100))

            input_mapping = mapping["indices"]
            input_positions = mapping["pos"]

        if processing_nside == out_nside:
            output_mapping = np.arange(coords_processing.shape[0])[:, np.newaxis]
            output_coordinates = None
        else:
            if out_nside is not None:
                output_coordinates = healpix_pixel_lonlat_torch(out_nside)
            else:
                assert(out_grid is not None)
                output_coordinates = get_coords_as_tensor(xr.open_dataset(out_grid), grid_type='regular') # change for other grids
            mapping = get_mapping_to_healpix_grid(coords_processing,
                                                  output_coordinates,
                                                  search_radius=search_radius,
                                                  max_nh=nh_input,
                                                  lowest_level=0,
                                                  periodic_fov=None)

            output_mapping = mapping["indices"]


        self.input_mapping = input_mapping
        self.input_positions = input_positions
        self.output_mapping = output_mapping
        self.input_coordinates = input_coordinates
        self.output_coordinates = output_coordinates

        if bottleneck_nside is None:
            self.data_input_mapping = input_mapping
            self.data_output_mapping = output_mapping
        else:
            self.data_input_mapping = self.data_output_mapping = np.arange(hp.nside2npix(bottleneck_nside))[:, np.newaxis]



        global_indices = np.arange(coords_processing.shape[0])
        data_global_indices = global_indices if bottleneck_nside is None else np.arange(hp.nside2npix(bottleneck_nside))

        if coarsen_sample_level == -1:
            self.global_cells = global_indices.reshape(1, -1)
            self.data_global_cells = data_global_indices.reshape(1, -1)
            self.global_cells_input = np.array([1]).reshape(-1, 1)
        else:
            self.global_cells = global_indices.reshape(-1, 4 ** coarsen_sample_level)
            self.data_global_cells = data_global_indices.reshape(self.global_cells.shape[0], -1)
            self.global_cells_input = self.global_cells[:, 0]

        ds_source = xr.open_dataset(self.files_source[0], decode_times=False)
        nt = self.n_sample_timesteps
        self.len_dataset = (len(self.sample_timesteps) - nt + 1)*self.global_cells.shape[0] if self.sample_timesteps else (ds_source["time"].shape[0] - nt + 1) * self.global_cells.shape[0]

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


    def get_data(self, ds, ts, global_indices, variables, global_level_start, index_mapping_dict=None):
        
        if index_mapping_dict is not None:
            indices = index_mapping_dict[global_indices // 4**global_level_start]
        else:
            indices = global_indices.reshape(-1,1)

        n, nh = indices.shape
        nt = self.n_sample_timesteps
        indices = indices.reshape(-1)

        # tbd: spatial dim as input!
        regular = False
        ts = torch.arange(ts, ts + nt, 1)
        if 'ncells' in dict(ds.sizes).keys():
            ds = ds.isel(ncells=indices, time=ts)
        elif 'cell' in dict(ds.sizes).keys():
            ds = ds.isel(cell=indices, time=ts)
        else:
            ds = ds.isel(time=ts)
            regular = True

        data_g = []
        for variable in variables:
            data = torch.tensor(ds[variable].values)
            data_g.append(data)
        data_g = torch.stack(data_g, dim=2)

        if regular:
            data_g = data_g.view(nt, len(variables), -1, data_g.shape[-1])
            data_g = data_g[:, :, indices]

        data_g = data_g.view(nt, -1, nh, len(variables), data_g.shape[-1] if data_g.dim() == 4 else 1)

        data_t = torch.tensor(ds["time"].values)

        ds.close()

        return data_g, data_t

    def __getitem__(self, index):
        source_index = 0
        source_file = self.files_source[source_index]
        
        target_file = self.files_target[source_index]

        
        ds_source, ds_target = self.get_files(source_file, file_path_target=target_file)

        if self.random_time_idx and not self.sample_timesteps:
            time_index = int(torch.randint(0, len(ds_source.time.values) - self.n_sample_timesteps + 1, (1, 1)))
        elif self.sample_timesteps:
            time_index = self.sample_timesteps[index // self.global_cells_input.shape[0]]

        global_cells_input = self.global_cells_input
        input_mapping = self.input_mapping
        global_cells = self.global_cells
        output_mapping = self.output_mapping

        if self.deterministic:
            region_index = torch.tensor(index % global_cells_input.shape[0])
        else:
            region_index = torch.randint(global_cells_input.shape[0],(1,))[0]
        if self.n_sample_vars != -1 and self.n_sample_vars != len(self.variables_source):
            sample_vars = torch.randperm(len(self.variables_source))[:self.n_sample_vars]
        else:
            sample_vars = torch.arange(len(self.variables_source))

        variables_source = np.array([self.variables_source[i.item()] for i in sample_vars])
        variables_target = np.array([self.variables_target[i.item()] for i in sample_vars])
        data_source, time_source = self.get_data(ds_source, time_index, self.data_global_cells[region_index], variables_source, 0, self.data_input_mapping)

        if ds_target is not None:
            data_target, time_target = self.get_data(ds_target, time_index, self.data_global_cells[region_index], variables_target, 0, self.data_output_mapping)
        else:
            data_target, time_target = data_source, time_source

        if self.input_coordinates is not None:
            coords_input = self.input_coordinates[input_mapping[global_cells[region_index]]]
        else:
            coords_input = torch.tensor([])

        if self.input_positions is not None:
            dists_input = self.input_positions[0][global_cells[region_index]]
        else:
            dists_input = torch.tensor([])

        if self.output_coordinates is not None:
            coords_output = self.output_coordinates[output_mapping[global_cells[region_index]]]
        else:
            coords_output = torch.tensor([])

        nt, n, nh, nv, f = data_source.shape

        drop_vars = torch.rand(1) < self.p_drop_vars
        drop_timesteps = torch.rand(1) < self.p_drop_timesteps
        drop_mask = torch.zeros((nt, n, nh, nv), dtype=bool)

        if torch.is_tensor(self.steady_mask):
            drop_mask = self.steady_mask[global_cells[region_index]].view(1, n, 1, 1).repeat(nt, 1, nh, nv)
        elif self.p_average > 0 and torch.rand(1) < self.p_average:
            avg_level = int(torch.randint(1, self.max_average_lvl + 1, (1,)))
            data_source_resh = data_source.view(-1,nt, 4 ** avg_level, nh, f)
            data_source_resh = data_source_resh.mean(dim=[2, 3], keepdim=True)
            data_source_resh = data_source_resh.repeat_interleave(4 ** avg_level, dim=2)
            data_source_resh = data_source_resh.repeat_interleave(nh, dim=3)
            data_source = data_source_resh.view(nt,n, nh, nv, f)

            if self.p_average_dropout > 0:
                if drop_vars:
                    drop_mask_p = (torch.rand((n // 4 ** avg_level, nv)) < self.p_average_dropout).bool()
                else:
                    drop_mask_p = (torch.rand((n // 4 ** avg_level)) < self.p_average_dropout).bool()

                drop_mask = drop_mask.view(nt, -1, 4 ** avg_level, nh, nv).transpose(-1, 2)
                drop_mask[drop_mask_p] = True
                drop_mask = drop_mask.transpose(-1, 2).view(nt, -1, nh, nv)
        else:
            if self.random_p and drop_vars:
                p_dropout = skewed_random_p(nv, exponent=self.skewness_exp, max_p=self.p_dropout)
            elif self.random_p:
                p_dropout = skewed_random_p(1, exponent=self.skewness_exp, max_p=self.p_dropout)
            else:
                p_dropout = torch.tensor(self.p_dropout)

            if self.p_dropout > 0 and not drop_vars and not drop_timesteps:
                drop_mask_p = (torch.rand((nt, n, nh)) < p_dropout).bool()
                drop_mask[drop_mask_p] = True

            elif self.p_dropout > 0 and drop_vars:
                drop_mask_p = (torch.rand((nt, n, nh, nv)) < p_dropout).bool()
                drop_mask[drop_mask_p] = True

            elif self.p_dropout > 0 and drop_timesteps:
                drop_mask_p = (torch.rand(nt)<self.p_dropout).bool()
                drop_mask[drop_mask_p]=True

        if self.n_drop_vars!=-1 and self.n_drop_vars < nv:
            not_drop_vars = torch.randperm(nv)[:(nv-self.n_drop_vars)]
            drop_mask[:,:,not_drop_vars] = (drop_mask[:,:,not_drop_vars]*0).bool()

        if self.norm_dict and self.bottleneck_nside is None:
            for k, var in enumerate(variables_source):
                data_source[:,:,:,k,:] = self.var_normalizers[var].normalize(data_source[:,:,:,k,:])

            for k, var in enumerate(variables_target):
                data_target[:,:,:,k,:] = self.var_normalizers[var].normalize(data_target[:,:,:,k,:])
        data_source[drop_mask] = 0

        if self.coarsen_sample_level == -1:
            indices_sample = torch.tensor([])
        else:
            indices_sample = {'sample': region_index.repeat(nt),
                              'sample_level': torch.tensor(self.coarsen_sample_level).repeat(nt)}
        var_indices = torch.tensor([np.where(np.array(self.variables_source_train)==var)[0][0] for var in variables_source])
        embed_data = {'VariableEmbedder': var_indices.unsqueeze(0).repeat(nt, 1),
                      'TimeEmbedder': time_source.float()}
        return data_source.float(), data_target.float(), coords_input.unsqueeze(0).repeat(nt, *[1] * coords_input.ndim).float(), coords_output.unsqueeze(0).repeat(nt, *[1] * coords_output.ndim).float(), indices_sample, drop_mask, embed_data, dists_input.unsqueeze(0).repeat(nt, *[1] * dists_input.ndim)

    def __len__(self):
        return self.len_dataset