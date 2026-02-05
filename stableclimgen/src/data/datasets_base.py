import copy
import json

import numpy as np
import torch
import xarray as xr
from omegaconf import ListConfig
from einops import rearrange
from torch.utils.data import Dataset

import warnings
warnings.filterwarnings("ignore", message=".*fails while guessing")

from ..modules.grids.grid_utils import get_coords_as_tensor,get_grid_type_from_var,get_mapping_weights,to_zoom, apply_zoom_diff, decode_zooms
from ..utils import normalizer as normalizers

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
                 p_dropout_all=0,
                 p_drop_groups=0,
                 n_drop_groups=-1,
                 random_p = False,
                 skewness_exp = 2,
                 n_sample_variables=-1,
                 deterministic=False,
                 output_binary_mask=False,
                 output_differences=True,
                 apply_diff = True,
                 output_max_zoom_only = False,
                 normalize_data = True,
                 mask_ts_mode = 'repeat'
                 ):
        
        super(BaseDataset, self).__init__()
        
        self.norm_dict = norm_dict
        self.lazy_load = lazy_load
        self.random_p = random_p
        self.p_dropout = p_dropout
        self.skewness_exp = skewness_exp
        self.n_sample_variables = n_sample_variables
        self.deterministic = deterministic
        self.p_drop_groups = p_drop_groups
        self.n_drop_groups = n_drop_groups
        self.output_differences = output_differences
        self.output_binary_mask = output_binary_mask
        self.mask_zooms = mask_zooms
        self.apply_diff = apply_diff
        self.output_max_zoom_only = output_max_zoom_only
        
        self.mask_ts_mode = mask_ts_mode
        self.p_dropout_all_zooms = dict(zip(self.sampling_zooms.keys(), [v.get("p_drop", 0) for v in self.sampling_zooms.values()]))
        self.mask_last_ts_zooms = dict(zip(self.sampling_zooms.keys(), [v.get("mask_last_ts", False) for v in self.sampling_zooms.values()]))

        self.p_dropout_all = p_dropout_all


        if "files" in self.data_dict['source'].keys():
            all_files = self.data_dict['source']["files"]

        all_files = []
        for data in self.data_dict['source'].values():
            if isinstance(data['files'], list) or isinstance(data['files'], ListConfig):
                all_files += data['files']
            else:
                all_files.append(data['files'])
        
        self.zoom_patch_sample = [v['zoom_patch_sample'] for v in self.sampling_zooms.values()]
        self.zoom_time_steps_past = [v['n_past_ts'] for v in self.sampling_zooms.values()]
        self.zoom_time_steps_future = [v['n_future_ts'] for v in self.sampling_zooms.values()]
        self.zooms = [z for z in self.sampling_zooms.keys()]
        
        self.max_time_step_past = max(self.zoom_time_steps_past)
        self.max_time_step_future = max(self.zoom_time_steps_future)

        #TODO fix the variable ids if multi var training/inference
      #  self.variables_source_train = variables_source_train if variables_source_train is not None else self.variables_source
       # self.variables_target_train = variables_target_train if variables_target_train is not None else self.variables_target

    #    self.var_indices = dict(zip(self.variables_source_train, np.arange(len(self.variables_source_train))))
        unique_time_steps_past = len(torch.tensor(self.zoom_time_steps_past).unique())==1
        unique_time_steps_future = len(torch.tensor(self.zoom_time_steps_future).unique())==1
        unique_zoom_patch_sample = len(torch.tensor(self.zoom_patch_sample).unique())==1

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
        for file_idx, num_timesteps in enumerate(self.time_steps_files):
            num_timesteps -= self.max_time_step_future

            start_idx = self.max_time_step_past

            for time_idx in range(start_idx, num_timesteps):
                if self.sample_timesteps is None or time_idx in self.sample_timesteps:
                    for zoom in self.zooms:
                        for region_idx_max in range(self.indices[max(self.zooms)].shape[0]):
                            if self.sampling_zooms[zoom]['zoom_patch_sample'] == -1:
                                region_idx_zoom = 0
                            else:
                                region_idx_zoom = region_idx_max//4**(self.sampling_zooms[max(self.zooms)]['zoom_patch_sample'] - self.sampling_zooms[zoom]['zoom_patch_sample'])

                            self.index_map[zoom].append((file_idx, time_idx, region_idx_zoom))
                
        self.index_map = {z: np.array(idx_map) for z, idx_map in self.index_map.items()}

        all_variables = []
        variable_ids = {}
        all_ids = []
        self.group_ids = {}
        offset = 0
        for group_id, (group, vars) in enumerate(self.data_dict['variables'].items()):
            all_variables += vars
            variable_ids[group] = np.arange(len(vars)) + offset
            all_ids = all_ids+list(variable_ids[group])
            offset = len(variable_ids[group])
            self.group_ids[group] = (group_id)
        
        
        self.all_variable_ids = dict(zip(all_variables, all_ids))

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

        self.load_once = unique_time_steps_past and unique_time_steps_future and unique_zoom_patch_sample and self.single_source

        with open(norm_dict) as json_file:
            norm_dict = json.load(json_file)

        self.var_normalizers = {}
        for zoom in self.zooms:
            self.var_normalizers[zoom] = {}
            for var in all_variables:
                if str(zoom) in norm_dict[var].keys():
                    norm_class = norm_dict[var][str(zoom)]['normalizer']['class']
                    assert norm_class in normalizers.__dict__.keys(), f'normalizer class {norm_class} not defined'
                    self.var_normalizers[zoom][var] = normalizers.__getattribute__(norm_class)(
                        norm_dict[var][str(zoom)]['stats'],
                        norm_dict[var][str(zoom)]['normalizer'])
                else:
                    norm_class = norm_dict[var]['normalizer']['class']
                    assert norm_class in normalizers.__dict__.keys(), f'normalizer class {norm_class} not defined'
                    self.var_normalizers[zoom][var] = normalizers.__getattribute__(norm_class)(
                        norm_dict[var]['stats'],
                        norm_dict[var]['normalizer'])
        self.normalize_data = normalize_data
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

    def _select_time_patch(self, ds, time_idx, patch_idx, mapping, mapping_zoom, zoom):
        # Fetch raw patch indices
        n_past_timesteps = self.sampling_zooms[zoom]['n_past_ts']
        n_future_timesteps = self.sampling_zooms[zoom]['n_future_ts']

        time_indices = np.arange(time_idx-n_past_timesteps, time_idx + n_future_timesteps + 1, 1)
        isel_dict = {"time": time_indices}
        patch_dim = [d for d in ds.dims if "cell" in d or "ncells" in d]
        patch_dim = patch_dim[0] if patch_dim else None

        nt = 1 + n_past_timesteps + n_future_timesteps

        for grid_type, variables_grid_type in self.grid_types_vars.items():

            mapping = mapping[grid_type]

            patch_indices = self.get_indices_from_patch_idx(zoom, patch_idx)

            post_map = mapping_zoom > zoom
            if post_map:
                indices = mapping['indices'][..., [0]].reshape(-1, 4 ** (mapping_zoom - zoom))
                if patch_dim:
                    isel_dict[patch_dim] = indices.view(-1)

            else:
                indices = mapping['indices'][..., [0]]

                if patch_dim:
                    isel_dict[patch_dim] = indices[patch_indices].view(-1)
            #print(isel_dict, ds.sizes)
            ds_zoom = ds.isel(isel_dict)
    
        return ds_zoom
    
    def get_data(self, ds, patch_idx, variables_sample, mapping, mapping_zoom, zoom, drop_mask=None):
        # Fetch raw patch indices
        drop_mask_ = drop_mask.clone() if drop_mask is not None else None
        if drop_mask_ is not None and drop_mask_.ndim == 2:
            drop_mask_ = drop_mask_.unsqueeze(0)

        n_past_timesteps = self.sampling_zooms[zoom]['n_past_ts']
        n_future_timesteps = self.sampling_zooms[zoom]['n_future_ts']

        patch_dim = [d for d in ds.dims if "cell" in d or "ncells" in d]
        patch_dim = patch_dim[0] if patch_dim else None

        nt = 1 + n_past_timesteps + n_future_timesteps

        data_g = []
        data_time_zoom = None
        for grid_type, variables_grid_type in self.grid_types_vars.items():
            variables = [var for var in variables_sample if var in variables_grid_type]
            if not variables:
                continue

            mapping = mapping[grid_type]

            patch_indices = self.get_indices_from_patch_idx(zoom, patch_idx)

            mask = (1. * get_mapping_weights(mapping)[..., 0]).view(1, 1, -1, 1, 1)

            post_map = mapping_zoom > zoom
            if post_map:
                indices = mapping['indices'][..., [0]].reshape(-1, 4 ** (mapping_zoom - zoom))

            else:
                indices = mapping['indices'][..., [0]]
                mask = mask[:, :, patch_indices]

                if drop_mask_ is not None:
                    drop_mask_ = drop_mask_[:, :, patch_indices]

            if data_time_zoom is None:
                data_time_zoom = torch.tensor(ds["time"].values).float()

            for variable in variables:
                values = ds[variable].values
                if values.ndim == 2:
                    values = values.reshape(nt, 1, -1)

                d = values.shape[1]

                if not patch_dim:
                    values = values.reshape(nt, d, -1)[:, :, indices.view(-1) if post_map else indices[patch_indices].view(-1)]
                data = torch.tensor(values).view(nt, d, -1)
                if self.normalize_data:
                    data = self.var_normalizers[zoom][variable].normalize(data)
                
                data = data.transpose(-1, -2)
                data_g.append(data.unsqueeze(dim=-1))

        if not data_g:
            data_g = torch.empty((0, nt, 0, 0, 0))
        else:
            data_g = torch.stack(data_g, dim=0)

        if drop_mask_ is not None:
            mask = (1-1.*drop_mask_.unsqueeze(dim=-1)) * mask
        
        data_g, mask = to_zoom(data_g, mapping_zoom, zoom, mask=mask.expand_as(data_g), binarize_mask=self.output_binary_mask)

        if post_map:
            data_g = data_g[:, :, patch_indices]
            mask = mask[:, :, patch_indices]

        drop_mask = torch.logical_not(mask[...,[0]]) if mask.dtype==torch.bool else mask[...,[0]]

        data_time = data_time_zoom if data_time_zoom is not None else torch.tensor(ds["time"].values).float()

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
  
    def _finalize_group(self, source_zooms, target_zooms, time_zooms, mask_mapping_zooms, patch_index_zooms, hr_dopout):
        sample_configs = copy.deepcopy(self.sampling_zooms)
        if not source_zooms:
            for key, value in patch_index_zooms.items():
                if key in sample_configs:
                    sample_configs[key]['patch_index'] = value
            return {}, {}, {}, sample_configs

        if self.apply_diff:
            data_source = apply_zoom_diff(source_zooms, sample_configs, patch_index_zooms)
            data_target = apply_zoom_diff(target_zooms, sample_configs, patch_index_zooms)
        else:
            data_source = source_zooms
            data_target = target_zooms

        if not hr_dopout and self.p_dropout_all > 0:
            drop = False
            for zoom in sorted(self.sampling_zooms.keys()):

                if self.p_dropout_all_zooms[zoom] > 0 and not drop:
                    drop = torch.rand(1) < self.p_dropout_all_zooms[zoom]

                if drop and mask_mapping_zooms[zoom].dtype == torch.bool:
                    mask_mapping_zooms[zoom] = torch.ones_like(mask_mapping_zooms[zoom], dtype=bool)

                elif drop:
                    mask_mapping_zooms[zoom] = torch.zeros_like(mask_mapping_zooms[zoom], dtype=mask_mapping_zooms[zoom].dtype)


        for zoom in data_source.keys():
            if mask_mapping_zooms[zoom].dtype == torch.float:
                mask_zoom = mask_mapping_zooms[zoom] == 0
            else:
                mask_zoom = mask_mapping_zooms[zoom]

            data_source[zoom][mask_zoom.expand_as(data_source[zoom])] = 0


        if any(['mask_last_ts' in z for z in self.sampling_zooms.values()]):
            drop = False
            for zoom, sampling_zoom in self.sampling_zooms.items():

                if sampling_zoom['mask_last_ts']:
                    mask_mapping_zooms[zoom][:,-1] = True

                    if self.mask_ts_mode == 'repeat':
                        data_source[zoom][:,-1] = data_source[zoom][:,-2]
                    else:
                        data_source[zoom][:,-1] = 0.

        

        for key, value in patch_index_zooms.items():
            if key in sample_configs:
                sample_configs[key]['patch_index'] = value

        if self.output_max_zoom_only:
            max_zoom = max(source_zooms.keys())
            data_source = decode_zooms(data_source, sample_configs=sample_configs, out_zoom=max_zoom)
            data_target = decode_zooms(data_target, sample_configs=sample_configs, out_zoom=max_zoom)

        return data_source, data_target, sample_configs, mask_mapping_zooms


    def __getitem__(self, index):
        
        selected_vars = {}
        selected_var_ids = {}
        var_indices = {}
        drop_indices = {}
        offset = 0
        group_keys = list(self.data_dict['variables'].keys())
        for group in group_keys:
            variables = self.data_dict['variables'][group]
            sample_size = len(variables) if self.n_sample_variables == -1 else min(self.n_sample_variables, len(variables))
            selected_vars[group] = np.random.choice(variables, sample_size, replace=False)
            selected_var_ids[group] = [self.all_variable_ids[str(variable)] for variable in selected_vars[group]]
            var_indices[group] = np.arange(len(selected_vars[group]))
            drop_indices[group] = var_indices[group] + offset
            offset += len(selected_vars[group])

        hr_dopout = self.p_dropout > 0 and torch.rand(1) > (self.p_dropout_all)

        if self.single_source and hr_dopout:
            nt = 1 + self.max_time_step_future + self.max_time_step_past
            total_vars = sum(len(selected_vars[group]) for group in selected_vars.keys())
            drop_mask_input = self.get_mask(total_vars, 
                                            nt, 
                                            self.indices[max(self.zooms)].size,
                                            self.p_dropout,
                                            self.p_drop_groups,
                                            self.n_drop_groups)
        else:
            drop_mask_input = None
            if self.p_dropout > 0:
                UserWarning('Multi-source input does not support global dropout')

        source_zooms_groups = [ {} for _ in group_keys ]
        target_zooms_groups = [ {} for _ in group_keys ]
        time_zooms = {}
        mask_mapping_zooms_groups = [ {} for _ in group_keys ]

        loaded = False
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

            if not loaded:
                ds_source, ds_target = self.get_files(source_file, file_path_target=target_file, drop_source=self.p_dropout>0)
                loaded = True if self.load_once else False

            if drop_mask_input is not None:
                ts_start = self.max_time_step_past - self.sampling_zooms[zoom]['n_past_ts']
                ts_end = self.max_time_step_future - self.sampling_zooms[zoom]['n_future_ts']
                drop_mask_zoom = drop_mask_input[:, ts_start:(drop_mask_input.shape[1] - ts_end)]
            else:
                drop_mask_zoom = None

            drop_mask_zoom_groups = []
            if drop_mask_zoom is None:
                drop_mask_zoom_groups = [None for _ in group_keys]
            else:
                for group in group_keys:
                    drop_mask_zoom_groups.append(drop_mask_zoom[drop_indices[group]].unsqueeze(0))

            data_time_zoom = None

            ds_source_zoom = self._select_time_patch(ds_source,
                    time_index,
                    patch_index,
                    self.mapping[mapping_zoom],
                    mapping_zoom,
                    zoom)
            
            if ds_target is not None:
                    ds_target_zoom = self._select_time_patch(
                        ds_target,
                        time_index,
                        patch_index,
                        self.mapping[mapping_zoom],
                        mapping_zoom,
                        zoom
                    )
            

            for group_idx, group in enumerate(group_keys):
                data_source, data_time_zoom_group, drop_mask_zoom_group = self.get_data(
                    ds_source_zoom,
                    patch_index,
                    selected_vars[group],
                    self.mapping[mapping_zoom],
                    mapping_zoom,
                    zoom,
                    drop_mask=drop_mask_zoom_groups[group_idx],
                )

                if ds_target is not None:
                    data_target, _, _ = self.get_data( #TODO ds_target is not sliced
                        ds_target_zoom,
                        patch_index,
                        selected_vars[group],
                        self.mapping[mapping_zoom],
                        mapping_zoom,
                        zoom,
                    )
                else:
                    data_target = data_source.clone()

                source_zooms_groups[group_idx][zoom] = data_source
                target_zooms_groups[group_idx][zoom] = data_target
                mask_mapping_zooms_groups[group_idx][zoom] = drop_mask_zoom_group

                if data_time_zoom is None:
                    data_time_zoom = data_time_zoom_group

            if data_time_zoom is not None:
                time_zooms[zoom] = data_time_zoom

    
        patch_index_zooms = {}
        for zoom in self.sampling_zooms.keys():
            patch_index_zooms[zoom] = torch.tensor(self.index_map[zoom][index][-1])

        source_zooms_groups_out = []
        target_zooms_groups_out = []
        mask_zooms_groups = []
        emb_groups = []

        emb = {}
        StaticVariableEmbedder = None
       # emb['DensityEmbedder'] = torch.tensor([selected_var_ids[group] for group in group_keys])
        for group_idx, group in enumerate(group_keys):
            if group == 'embedding': 
                StaticVariableEmbedder = source_zooms_groups[group_idx]
                StaticVariableEmbedder = dict(zip(StaticVariableEmbedder.keys(), [rearrange(t, 'v t n f d-> t n (v f d)') for t in StaticVariableEmbedder.values()]))

        for group_idx, group in enumerate(group_keys):
            if group != 'embedding': 
                source_zooms, target_zooms, _, mask_group = self._finalize_group(
                    source_zooms_groups[group_idx],
                    target_zooms_groups[group_idx],
                    time_zooms,
                    mask_mapping_zooms_groups[group_idx],
                    patch_index_zooms,
                    hr_dopout,
                )
                source_zooms_groups_out.append(source_zooms)
                target_zooms_groups_out.append(target_zooms)
                mask_zooms_groups.append(mask_group)

                emb_group = emb.copy()
                emb_group['VariableEmbedder'] = torch.tensor(list(var_indices[group])).view(-1)
                emb_group['MGEmbedder'] = emb_group['VariableEmbedder']

                if StaticVariableEmbedder is not None:
                    emb_group['StaticVariableEmbedder'] = StaticVariableEmbedder

                emb_group['TimeEmbedder'] = time_zooms
                emb_groups.append(emb_group)

        return source_zooms_groups_out, target_zooms_groups_out, mask_zooms_groups, emb_groups, patch_index_zooms

    def __len__(self):
        return self.len_dataset
