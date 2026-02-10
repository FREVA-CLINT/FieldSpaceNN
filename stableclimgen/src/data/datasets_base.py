import copy
import json
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

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

def skewed_random_p(
    size: Union[int, Sequence[int], torch.Size],
    exponent: float = 2,
    max_p: float = 0.9,
) -> torch.Tensor:
    """
    Generate a skewed random probability tensor.

    :param size: Output size for the generated tensor (any torch.Size-compatible shape).
    :param exponent: Exponent controlling the skew of the distribution.
    :param max_p: Maximum probability value.
    :return: Tensor of shape ``size`` with values in ``[0, max_p]``.
    """
    uniform_random: torch.Tensor = torch.rand(size)
    skewed_random: torch.Tensor = max_p * (1 - uniform_random ** exponent)
    return skewed_random

def invert_dict(d: Mapping[Any, Any]) -> Dict[Any, List[Any]]:
    """
    Invert a dictionary mapping values to lists of keys.

    :param d: Input mapping to invert.
    :return: Dictionary that groups original keys by their values.
    """
    inverted_d: Dict[Any, List[Any]] = {}
    for key, value in d.items():
        inverted_d.setdefault(value, []).append(key)
    return inverted_d

#def create_mask(random_p, drop_mask, ):

class BaseDataset(Dataset):
    def __init__(
        self,
        mapping_fcn: Optional[Callable[..., Any]] = None,
        norm_dict: Optional[str] = None,
        lazy_load: bool = True,
        mask_zooms: Optional[Mapping[int, Any]] = None,
        p_dropout: float = 0,
        p_dropout_all: float = 0,
        p_drop_groups: float = 0,
        n_drop_groups: int = -1,
        random_p: bool = False,
        skewness_exp: float = 2,
        n_sample_variables: int = -1,
        deterministic: bool = False,
        output_binary_mask: bool = False,
        output_differences: bool = True,
        apply_diff: bool = True,
        output_max_zoom_only: bool = False,
        normalize_data: bool = True,
        mask_ts_mode: str = 'repeat',
        variables_as_features: bool = False,
        load_n_samples_time: int = 1,
    ) -> None:
        """
        Initialize the dataset with sampling, masking, and normalization settings.

        :param mapping_fcn: Callable to build mapping weights between grids.
        :param norm_dict: Path to the JSON normalization statistics file.
        :param lazy_load: Whether to lazily load xarray datasets.
        :param mask_zooms: Optional mask configuration per zoom level.
        :param p_dropout: Base dropout probability for spatial masking.
        :param p_dropout_all: Probability to drop entire samples across zooms.
        :param p_drop_groups: Probability to drop whole variable groups.
        :param n_drop_groups: Number of variable groups to keep (or -1 for all).
        :param random_p: Whether to sample dropout probabilities per variable.
        :param skewness_exp: Exponent used for skewed dropout sampling.
        :param n_sample_variables: Number of variables to sample per group (-1 for all).
        :param deterministic: Whether to use deterministic sampling behavior.
        :param output_binary_mask: Whether to output binary masks.
        :param output_differences: Whether to output temporal differences.
        :param apply_diff: Whether to apply zoom-difference transforms.
        :param output_max_zoom_only: Whether to output only the max zoom results.
        :param normalize_data: Whether to normalize variables with saved stats.
        :param mask_ts_mode: Strategy for masking the last timestep.
        :param variables_as_features: Whether to treat variables as features.
        :param load_n_samples_time: Number of time samples stacked as batch.
        :return: None.
        """
        super(BaseDataset, self).__init__()

        self.norm_dict: Optional[str] = norm_dict
        self.lazy_load: bool = lazy_load
        self.random_p: bool = random_p
        self.p_dropout: float = p_dropout
        self.skewness_exp: float = skewness_exp
        self.n_sample_variables: int = n_sample_variables
        self.deterministic: bool = deterministic
        self.p_drop_groups: float = p_drop_groups
        self.n_drop_groups: int = n_drop_groups
        self.output_differences: bool = output_differences
        self.output_binary_mask: bool = output_binary_mask
        self.mask_zooms: Optional[Mapping[int, Any]] = mask_zooms
        self.apply_diff: bool = apply_diff
        self.output_max_zoom_only: bool = output_max_zoom_only
        self.variables_as_features: bool = variables_as_features

        self.load_n_samples_time: int = load_n_samples_time

        self.mask_ts_mode: str = mask_ts_mode
        self.p_dropout_all_zooms: Dict[int, float] = dict(
            zip(self.sampling_zooms.keys(), [v.get("p_drop", 0) for v in self.sampling_zooms.values()])
        )
        self.mask_last_ts_zooms: Dict[int, bool] = dict(
            zip(self.sampling_zooms.keys(), [v.get("mask_last_ts", False) for v in self.sampling_zooms.values()])
        )

        self.p_dropout_all: float = p_dropout_all


        if "files" in self.data_dict['source'].keys():
            all_files: List[str] = self.data_dict['source']["files"]

        all_files = []
        for data in self.data_dict['source'].values():
            if isinstance(data['files'], list) or isinstance(data['files'], ListConfig):
                all_files += data['files']
            else:
                all_files.append(data['files'])
        
        self.zoom_patch_sample: List[int] = [v['zoom_patch_sample'] for v in self.sampling_zooms.values()]
        self.zoom_time_steps_past: List[int] = [v['n_past_ts'] for v in self.sampling_zooms.values()]
        self.zoom_time_steps_future: List[int] = [v['n_future_ts'] for v in self.sampling_zooms.values()]
        self.zooms: List[int] = [z for z in self.sampling_zooms.keys()]
        
        self.max_time_step_past: int = max(self.zoom_time_steps_past)
        self.max_time_step_future: int = max(self.zoom_time_steps_future)

        #TODO fix the variable ids if multi var training/inference
      #  self.variables_source_train = variables_source_train if variables_source_train is not None else self.variables_source
       # self.variables_target_train = variables_target_train if variables_target_train is not None else self.variables_target

    #    self.var_indices = dict(zip(self.variables_source_train, np.arange(len(self.variables_source_train))))
        unique_time_steps_past: bool = len(torch.tensor(self.zoom_time_steps_past).unique()) == 1
        unique_time_steps_future: bool = len(torch.tensor(self.zoom_time_steps_future).unique()) == 1
        unique_zoom_patch_sample: bool = len(torch.tensor(self.zoom_patch_sample).unique()) == 1

        if "timesteps" in self.data_dict.keys():
            self.sample_timesteps: List[int] = []
            for t in self.data_dict["timesteps"]:
                if isinstance(t, int) or "-" not in t:
                    self.sample_timesteps.append(int(t))
                else:
                    start, end = map(int, t.split("-"))
                    self.sample_timesteps += list(range(start, end))
            self.sample_timesteps = self.sample_timesteps
        else:
            self.sample_timesteps: Optional[List[int]] = None

        self.time_steps_files: List[int] = []
        for k, file in enumerate(self.data_dict['source'][self.zooms[0]]['files']):
            ds: xr.Dataset = xr.open_dataset(file)
            self.time_steps_files.append(len(ds.time))

        # Build a lookup from zoom to available (file, time window, region) samples.
        self.index_map: Dict[int, List[Tuple[int, List[int], int]]] = dict(
            zip(self.zooms, [[] for _ in self.zooms])
        )
        for file_idx, num_timesteps in enumerate(self.time_steps_files):
            file_idx: int
            num_timesteps: int
            num_timesteps -= self.max_time_step_future

            start_idx: int = self.max_time_step_past
            if self.sample_timesteps is None:
                time_indices: List[int] = list(range(start_idx, num_timesteps))
            else:
                time_indices: List[int] = [t for t in self.sample_timesteps if start_idx <= t < num_timesteps]

            time_entries: np.ndarray = np.array(time_indices).reshape(-1, self.load_n_samples_time)

            for time_entry in time_entries:
                time_entry: np.ndarray
                for zoom in self.zooms:
                    for region_idx_max in range(self.indices[max(self.zooms)].shape[0]):
                        region_idx_max: int
                        region_idx_zoom: int
                        if self.sampling_zooms[zoom]['zoom_patch_sample'] == -1:
                            region_idx_zoom = 0
                        else:
                            region_idx_zoom = region_idx_max//4**(self.sampling_zooms[max(self.zooms)]['zoom_patch_sample'] - self.sampling_zooms[zoom]['zoom_patch_sample'])

                        self.index_map[zoom].append((file_idx, list(time_entry), region_idx_zoom))
                
        #self.index_map = {z: np.array(idx_map) for z, idx_map in self.index_map.items()}

        # Build variable group indices for embedding and masking.
        all_variables: List[str] = []
        variable_ids: Dict[str, np.ndarray] = {}
        all_ids: List[int] = []
        self.group_ids: Dict[str, int] = {}
        offset: int = 0
        for group_id, (group, vars) in enumerate(self.data_dict['variables'].items()):
            all_variables += vars
            variable_ids[group] = np.arange(len(vars)) + offset
            all_ids = all_ids+list(variable_ids[group])
            offset = len(variable_ids[group])
            self.group_ids[group] = (group_id)
        
        self.all_variable_ids: Dict[str, int] = dict(zip(all_variables, all_ids))

        grid_types: List[Any] = [get_grid_type_from_var(ds, var) for var in all_variables]
        self.vars_grid_types: Dict[str, Any] = dict(zip(all_variables, grid_types))
        self.grid_types: np.ndarray = np.unique(grid_types)

        self.grid_types_vars: Dict[Any, List[str]] = invert_dict(self.vars_grid_types)
        for var, gtype in zip(all_variables, grid_types):
            self.grid_types_vars[gtype].append(var)

        unique_files: np.ndarray = np.unique(np.array(all_files))
        
        if len(unique_files)==1:
            self.single_source: bool = True
            # Single-source: build a shared mapping at the highest zoom and reuse across zooms.
            coords: List[torch.Tensor] = [
                get_coords_as_tensor(ds, grid_type=grid_type) for grid_type in self.grid_types
            ]
            mapping_hr: Dict[Any, Any] = dict(
                zip(self.grid_types, [mapping_fcn(coords_, max(self.zooms))[max(self.zooms)] for coords_ in coords])
            )
            self.mapping: Dict[int, Dict[Any, Any]] = {max(self.zooms): mapping_hr}
        else:
            self.single_source = False
            self.mapping = {}
            for zoom in self.zooms:
                # Multi-source: build a per-zoom mapping using that zoom's grid.
                mapping_grid_type: Dict[Any, Any] = {}
                for grid_type in self.grid_types:
                    ds: xr.Dataset = xr.open_dataset(self.data_dict['source'][zoom]['files'][0])
                    coords: torch.Tensor = get_coords_as_tensor(ds, grid_type=grid_type)
                    mapping_grid_type[grid_type] = mapping_fcn(coords, zoom)[zoom]
                self.mapping[zoom] = mapping_grid_type

        self.load_once: bool = (
            unique_time_steps_past and unique_time_steps_future and unique_zoom_patch_sample and self.single_source
        )

        with open(norm_dict) as json_file:
            norm_dict: Dict[str, Any] = json.load(json_file)

        self.var_normalizers: Dict[int, Dict[str, Any]] = {}
        for zoom in self.zooms:
            self.var_normalizers[zoom] = {}
            for var in all_variables:
                if str(zoom) in norm_dict[var].keys():
                    # Zoom-specific stats override global stats when available.
                    norm_class: str = norm_dict[var][str(zoom)]['normalizer']['class']
                    assert norm_class in normalizers.__dict__.keys(), f'normalizer class {norm_class} not defined'
                    self.var_normalizers[zoom][var] = normalizers.__getattribute__(norm_class)(
                        norm_dict[var][str(zoom)]['stats'],
                        norm_dict[var][str(zoom)]['normalizer'])
                else:
                    norm_class: str = norm_dict[var]['normalizer']['class']
                    assert norm_class in normalizers.__dict__.keys(), f'normalizer class {norm_class} not defined'
                    self.var_normalizers[zoom][var] = normalizers.__getattribute__(norm_class)(
                        norm_dict[var]['stats'],
                        norm_dict[var]['normalizer'])
        self.normalize_data: bool = normalize_data
        self.len_dataset: int = len(list(self.index_map.values())[0])
    
    def get_indices_from_patch_idx(self, patch_idx: int) -> np.ndarray:
        """
        Resolve a patch index to the underlying pixel indices.

        :param patch_idx: Patch index within the sampling grid.
        :return: NumPy array of indices for the patch (shape ``(n,)`` or ``(n, 1)``).
        """
        raise NotImplementedError

    def get_files(
        self,
        file_path_source: str,
        file_path_target: Optional[str] = None,
        drop_source: bool = False,
    ) -> Tuple[xr.Dataset, Optional[xr.Dataset]]:
        """
        Load source and target datasets from disk.

        :param file_path_source: Path to the source dataset file.
        :param file_path_target: Optional path to the target dataset file.
        :param drop_source: Whether to skip loading target when sharing the source.
        :return: Tuple of (source dataset, target dataset or None).
        """
        if self.lazy_load:
            ds_source: xr.Dataset = xr.open_dataset(file_path_source, decode_times=False)
        else:
            ds_source = xr.load_dataset(file_path_source, decode_times=False)

        if file_path_target is None:
            ds_target: Optional[xr.Dataset] = copy.deepcopy(ds_source)

        elif file_path_target == file_path_source and not drop_source:
            ds_target = None
            
        else:
            if self.lazy_load:
                ds_target = xr.open_dataset(file_path_target, decode_times=False)
            else:
                ds_target = xr.load_dataset(file_path_target, decode_times=False)

        return ds_source, ds_target

    #def map_data(self):

    def select_ranges(
        self,
        ds: xr.Dataset,
        time_indices: Sequence[int],
        patch_idx: int,
        mapping: Mapping[str, Any],
        mapping_zoom: int,
        zoom: int,
    ) -> xr.Dataset:
        """
        Select a time window and spatial patch from a dataset.

        :param ds: Input xarray dataset to slice.
        :param time_indices: Center time indices to expand into a window.
        :param patch_idx: Patch index within the zoom grid.
        :param mapping: Mapping dictionary for grid transforms.
        :param mapping_zoom: Zoom level of the mapping source.
        :param zoom: Zoom level of the requested data.
        :return: Sliced xarray dataset for the requested range.
        """
        # Fetch raw patch indices
        n_past_timesteps: int = self.sampling_zooms[zoom]['n_past_ts']
        n_future_timesteps: int = self.sampling_zooms[zoom]['n_future_ts']

        time_indices: np.ndarray = np.stack(
            [
                np.arange(time_index - n_past_timesteps, time_index + n_future_timesteps + 1, 1)
                for time_index in time_indices
            ],
            axis=0,
        ).reshape(-1)

        isel_dict: Dict[str, Any] = {"time": time_indices}
        patch_dim_candidates: List[str] = [d for d in ds.dims if "cell" in d or "ncells" in d]
        patch_dim: Optional[str] = patch_dim_candidates[0] if patch_dim_candidates else None

        nt: int = 1 + n_past_timesteps + n_future_timesteps

        for grid_type, variables_grid_type in self.grid_types_vars.items():
            mapping = mapping[grid_type]
            patch_indices: np.ndarray = self.get_indices_from_patch_idx(zoom, patch_idx)

            post_map: bool = mapping_zoom > zoom
            if post_map:
                indices: torch.Tensor = mapping['indices'][..., [0]].reshape(-1, 4 ** (mapping_zoom - zoom))
                if patch_dim:
                    isel_dict[patch_dim] = indices.view(-1)

            else:
                indices: torch.Tensor = mapping['indices'][..., [0]]

                if patch_dim:
                    isel_dict[patch_dim] = indices[patch_indices].view(-1)
            #print(isel_dict, ds.sizes)
            ds_zoom: xr.Dataset = ds.isel(isel_dict)
    
        return ds_zoom
    
    def get_data(
        self,
        ds: xr.Dataset,
        patch_idx: int,
        variables_sample: Sequence[str],
        mapping: Mapping[str, Any],
        mapping_zoom: int,
        zoom: int,
        drop_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract data, time values, and masks for a given patch.

        :param ds: Input xarray dataset.
        :param patch_idx: Patch index within the zoom grid.
        :param variables_sample: Variables to extract for this sample.
        :param mapping: Mapping dictionary for grid transforms.
        :param mapping_zoom: Zoom level of the mapping source.
        :param zoom: Zoom level of the requested data.
        :param drop_mask: Optional dropout mask tensor of shape ``(v, t, n)`` or ``(1, v, t, n)``.
        :return: Tuple ``(data_g, data_time, drop_mask)`` where ``data_g`` is a tensor of
            shape ``(v, t, n, d, f)``, ``data_time`` is a tensor of shape ``(t,)``, and
            ``drop_mask`` is a tensor of shape ``(v, t, n, d, 1)``.
        """
        # Fetch raw patch indices
        drop_mask_: Optional[torch.Tensor] = drop_mask.clone() if drop_mask is not None else None
        if drop_mask_ is not None and drop_mask_.ndim == 2:
            drop_mask_ = drop_mask_.unsqueeze(0)

        n_past_timesteps: int = self.sampling_zooms[zoom]['n_past_ts']
        n_future_timesteps: int = self.sampling_zooms[zoom]['n_future_ts']

        patch_dim_candidates: List[str] = [d for d in ds.dims if "cell" in d or "ncells" in d]
        patch_dim: Optional[str] = patch_dim_candidates[0] if patch_dim_candidates else None

        nt: int = (1 + n_past_timesteps + n_future_timesteps) * self.load_n_samples_time

        data_g: List[torch.Tensor] = []
        data_time_zoom: Optional[torch.Tensor] = None
        for grid_type, variables_grid_type in self.grid_types_vars.items():
            variables: List[str] = [var for var in variables_sample if var in variables_grid_type]
            if not variables:
                continue

            mapping = mapping[grid_type]

            patch_indices: np.ndarray = self.get_indices_from_patch_idx(zoom, patch_idx)

            mask: torch.Tensor = (1. * get_mapping_weights(mapping)[..., 0]).view(1, 1, -1, 1, 1)

            post_map: bool = mapping_zoom > zoom
            if post_map:
                indices: torch.Tensor = mapping['indices'][..., [0]].reshape(-1, 4 ** (mapping_zoom - zoom))

            else:
                indices: torch.Tensor = mapping['indices'][..., [0]]
                mask = mask[:, :, patch_indices]

                if drop_mask_ is not None:
                    drop_mask_ = drop_mask_[:, :, patch_indices]

            if data_time_zoom is None:
                data_time_zoom = torch.tensor(ds["time"].values).float()

            for variable in variables:
                values: np.ndarray = ds[variable].values
                if values.ndim == 2:
                    values = values.reshape(nt, 1, -1)

                d: int = values.shape[1]

                if not patch_dim:
                    values = values.reshape(nt, d, -1)[:, :, :, indices.view(-1) if post_map else indices[patch_indices].view(-1)]

                data: torch.Tensor = torch.tensor(values).view(nt, d, -1)

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

        drop_mask: torch.Tensor = torch.logical_not(mask[...,[0]]) if mask.dtype==torch.bool else mask[...,[0]]

        data_time: torch.Tensor = data_time_zoom if data_time_zoom is not None else torch.tensor(ds["time"].values).float()

        return data_g, data_time, drop_mask

   #def get_masks_zooms(self, grid_type):

    def get_mask(
        self,
        ng: int,
        nt: int,
        n: int,
        p_dropout: float = 0,
        p_drop_groups: float = 0,
        p_drop_time_steps: float = 0,
        n_drop_groups: int = -1,
    ) -> torch.Tensor:
        """
        Build a dropout mask over variables, time, and space.

        :param ng: Number of variables (``v`` dimension).
        :param nt: Number of timesteps (``t`` dimension).
        :param n: Number of spatial points (``n`` dimension).
        :param p_dropout: Base dropout probability.
        :param p_drop_groups: Probability of dropping entire variable groups.
        :param p_drop_time_steps: Probability of dropping entire timesteps.
        :param n_drop_groups: Number of variable groups to keep (or -1 for all).
        :return: Boolean mask tensor of shape ``(v, t, n)``.
        """
        drop_groups: torch.Tensor = torch.rand(1) < p_drop_groups
        drop_timesteps: torch.Tensor = torch.rand(1) < p_drop_time_steps
        drop_mask: torch.Tensor = torch.zeros((ng, nt, n), dtype=bool)

        if self.random_p and drop_groups:
            p_dropout: torch.Tensor = skewed_random_p(ng, exponent=self.skewness_exp, max_p=p_dropout)
        elif self.random_p:
            p_dropout: torch.Tensor = skewed_random_p(1, exponent=self.skewness_exp, max_p=p_dropout)
        else:
            p_dropout: torch.Tensor = torch.tensor(p_dropout)

        if p_dropout > 0 and not drop_groups and not drop_timesteps:
            drop_mask_p: torch.Tensor = (torch.rand((nt, n)) < p_dropout).bool()
            drop_mask[:, drop_mask_p] = True

        elif p_dropout > 0 and drop_groups:
            drop_mask_p: torch.Tensor = (torch.rand((ng, nt, n)) < p_dropout.view(-1, 1, 1)).bool()
            drop_mask[drop_mask_p] = True

        elif p_dropout > 0 and drop_timesteps:
            drop_mask_p: torch.Tensor = (torch.rand(nt) < p_dropout).bool()
            drop_mask[:, drop_mask_p] = True

        if n_drop_groups != -1 and n_drop_groups < ng:
            not_drop_vars: torch.Tensor = torch.randperm(ng)[:(ng - n_drop_groups)]
            drop_mask[not_drop_vars] = (drop_mask[not_drop_vars] * 0).bool()

        return drop_mask
  
    def _finalize_group(
        self,
        data_source: Mapping[int, torch.Tensor],
        data_target: Mapping[int, torch.Tensor],
        time_zooms: Mapping[int, torch.Tensor],
        mask_mapping_zooms: Mapping[int, torch.Tensor],
        patch_index_zooms: Mapping[int, torch.Tensor],
        hr_dopout: bool,
    ) -> Tuple[Any, Any, Dict[int, Dict[str, Any]], Dict[int, torch.Tensor]]:
        """
        Finalize group data by applying masks, reshaping, and zoom transforms.

        :param data_source: Mapping from zoom to source tensor of shape ``(v, t, n, d, f)``.
        :param data_target: Mapping from zoom to target tensor of shape ``(v, t, n, d, f)``.
        :param time_zooms: Mapping from zoom to time tensor of shape ``(t,)`` or ``(b, t)``.
        :param mask_mapping_zooms: Mapping from zoom to mask tensor of shape ``(v, t, n, d, f)``.
        :param patch_index_zooms: Mapping from zoom to patch index tensor of shape ``(1,)``.
        :param hr_dopout: Whether high-resolution dropout is active.
        :return: Tuple ``(data_source, data_target, sample_configs, mask_mapping_zooms)`` where
            data tensors are reshaped to ``(b, v, t, n, d, f)`` (or ``(b, 1, t, n, 1, f)``
            when ``variables_as_features`` is enabled).
        """
        sample_configs: Dict[int, Dict[str, Any]] = copy.deepcopy(self.sampling_zooms)
        if not data_source:
            for key, value in patch_index_zooms.items():
                if key in sample_configs:
                    sample_configs[key]['patch_index'] = value
            return {}, {}, {}, sample_configs

        if not hr_dopout and self.p_dropout_all > 0:
            drop: Union[bool, torch.Tensor] = False
            for zoom in sorted(self.sampling_zooms.keys()):

                if self.p_dropout_all_zooms[zoom] > 0 and not drop:
                    drop = torch.rand(1) < self.p_dropout_all_zooms[zoom]

                if drop and mask_mapping_zooms[zoom].dtype == torch.bool:
                    mask_mapping_zooms[zoom] = torch.ones_like(mask_mapping_zooms[zoom], dtype=bool)

                elif drop:
                    mask_mapping_zooms[zoom] = torch.zeros_like(mask_mapping_zooms[zoom], dtype=mask_mapping_zooms[zoom].dtype)


        # Apply computed masks to zero-out dropped entries.
        for zoom in data_source.keys():
            if mask_mapping_zooms[zoom].dtype == torch.float:
                mask_zoom: torch.Tensor = mask_mapping_zooms[zoom] == 0
            else:
                mask_zoom = mask_mapping_zooms[zoom]

            data_source[zoom][mask_zoom.expand_as(data_source[zoom])] = 0


        for key, value in patch_index_zooms.items():
            if key in sample_configs:
                sample_configs[key]['patch_index'] = value

        if self.variables_as_features:
            # Collapse variables into the feature dimension.
            for zoom, x in data_source.items():
                data_source[zoom] = rearrange(x, 'v (b t) n d f -> b 1 t n 1 (v d f)', b=self.load_n_samples_time)
                data_target[zoom] = rearrange(data_target[zoom], 'v (b t) n d f -> b 1 t n 1 (v d f)', b=self.load_n_samples_time)
                mask_mapping_zooms[zoom] = rearrange(mask_mapping_zooms[zoom], 'v (b t) n d f -> b 1 t n 1 (v d f)', b=self.load_n_samples_time)
        else:
            for zoom, x in data_source.items():
                data_source[zoom] = rearrange(x, 'v (b t) n d f -> b v t n d f', b=self.load_n_samples_time)
                data_target[zoom] = rearrange(data_target[zoom], 'v (b t) n d f -> b v t n d f', b=self.load_n_samples_time)
                mask_mapping_zooms[zoom] = rearrange(mask_mapping_zooms[zoom], 'v (b t) n d f -> b v t n d f', b=self.load_n_samples_time)
        
        if self.output_max_zoom_only:
            max_zoom: int = max(data_source.keys())
            data_source = data_source[max_zoom]
            data_target = data_target[max_zoom]
        else:
            data_source = apply_zoom_diff(data_source, sample_configs, patch_index_zooms)
            data_target = apply_zoom_diff(data_target, sample_configs, patch_index_zooms)

        if any(['mask_last_ts' in z for z in self.sampling_zooms.values()]):
            drop: Union[bool, torch.Tensor] = False
            for zoom, sampling_zoom in self.sampling_zooms.items():

                if sampling_zoom['mask_last_ts']:
                    mask_mapping_zooms[zoom][:, :, -1] = True

                    if self.mask_ts_mode == 'repeat':
                        data_source[zoom][:, :, -1] = data_source[zoom][:, :, -2]
                    else:
                        data_source[zoom][:, :, -1] = 0.

        return data_source, data_target, sample_configs, mask_mapping_zooms


    def __getitem__(
        self,
        index: int,
    ) -> Tuple[List[Any], List[Any], List[Any], List[Dict[str, Any]], Dict[int, torch.Tensor]]:
        """
        Fetch a single dataset sample by index.

        :param index: Dataset index to retrieve.
        :return: Tuple ``(sources, targets, masks, embeddings, patch_index_zooms)`` where
            ``sources`` and ``targets`` are lists of per-group zoom mappings to tensors of
            shape ``(b, v, t, n, d, f)`` (or ``(b, 1, t, n, 1, f)`` when variables are folded
            into features), ``masks`` follow the same shape, ``embeddings`` holds per-group
            tensors such as ``VariableEmbedder`` of shape ``(v,)`` and ``TimeEmbedder`` of
            shape ``(b, t)``, and ``patch_index_zooms`` maps zoom to index tensors of shape
            ``(1,)``.
        """
        selected_vars: Dict[str, np.ndarray] = {}
        selected_var_ids: Dict[str, List[int]] = {}
        var_indices: Dict[str, np.ndarray] = {}
        drop_indices: Dict[str, np.ndarray] = {}
        offset: int = 0
        group_keys: List[str] = list(self.data_dict['variables'].keys())
        # Sample variables per group to build a compact input for this item.
        for group in group_keys:
            variables: List[str] = self.data_dict['variables'][group]
            sample_size: int = len(variables) if self.n_sample_variables == -1 else min(self.n_sample_variables, len(variables))
            selected_vars[group] = np.random.choice(variables, sample_size, replace=False)
            selected_var_ids[group] = [self.all_variable_ids[str(variable)] for variable in selected_vars[group]]
            var_indices[group] = np.arange(len(selected_vars[group]))
            drop_indices[group] = var_indices[group] + offset
            offset += len(selected_vars[group])

        hr_dopout: Union[bool, torch.Tensor] = self.p_dropout > 0 and torch.rand(1) > (self.p_dropout_all)

        if self.single_source and hr_dopout:
            nt: int = 1 + self.max_time_step_future + self.max_time_step_past
            total_vars: int = sum(len(selected_vars[group]) for group in selected_vars.keys())
            drop_mask_input: Optional[torch.Tensor] = self.get_mask(
                total_vars,
                nt,
                self.indices[max(self.zooms)].size,
                self.p_dropout,
                self.p_drop_groups,
                self.n_drop_groups,
            )
        else:
            drop_mask_input = None
            if self.p_dropout > 0:
                UserWarning('Multi-source input does not support global dropout')

        time_indices: List[int] = self.index_map[self.zooms[0]][index][1]

       
        source_zooms_groups: List[Dict[int, torch.Tensor]] = [{} for _ in group_keys]
        target_zooms_groups: List[Dict[int, torch.Tensor]] = [{} for _ in group_keys]
        time_zooms: Dict[int, torch.Tensor] = {}
        mask_mapping_zooms_groups: List[Dict[int, torch.Tensor]] = [{} for _ in group_keys]

        loaded: bool = False
        for zoom in self.zooms:
            file_index: int
            patch_index: int
            file_index, _, patch_index = self.index_map[zoom][index]
            if self.single_source:
                source_file: str = self.data_dict['source'][max(self.zooms)]['files'][int(file_index)]
                target_file: str = self.data_dict['target'][max(self.zooms)]['files'][int(file_index)]
                mapping_zoom: int = max(self.zooms)
                
            else:
                source_file: str = self.data_dict['source'][zoom]['files'][int(file_index)]
                target_file: str = self.data_dict['target'][zoom]['files'][int(file_index)]
                mapping_zoom: int = zoom

            if not loaded:
                ds_source: xr.Dataset
                ds_target: Optional[xr.Dataset]
                ds_source, ds_target = self.get_files(source_file, file_path_target=target_file, drop_source=self.p_dropout>0)
                loaded = True if self.load_once else False

            if drop_mask_input is not None:
                ts_start: int = self.max_time_step_past - self.sampling_zooms[zoom]['n_past_ts']
                ts_end: int = self.max_time_step_future - self.sampling_zooms[zoom]['n_future_ts']
                drop_mask_zoom: Optional[torch.Tensor] = drop_mask_input[:, ts_start:(drop_mask_input.shape[1] - ts_end)]
            else:
                drop_mask_zoom: Optional[torch.Tensor] = None

            drop_mask_zoom_groups: List[Optional[torch.Tensor]] = []
            if drop_mask_zoom is None:
                drop_mask_zoom_groups = [None for _ in group_keys]
            else:
                for group in group_keys:
                    drop_mask_zoom_groups.append(drop_mask_zoom[drop_indices[group]].unsqueeze(0))

            data_time_zoom: Optional[torch.Tensor] = None

            ds_source_zoom: xr.Dataset = self.select_ranges(
                ds_source,
                time_indices,
                patch_index,
                self.mapping[mapping_zoom],
                mapping_zoom,
                zoom,
            )
            
            if ds_target is not None:
                ds_target_zoom: xr.Dataset = self.select_ranges(
                    ds_target,
                    time_indices,
                    patch_index,
                    self.mapping[mapping_zoom],
                    mapping_zoom,
                    zoom,
                )
            

            for group_idx, group in enumerate(group_keys):
                data_source: torch.Tensor
                data_time_zoom_group: torch.Tensor
                drop_mask_zoom_group: torch.Tensor
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
                    data_target: torch.Tensor
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

        
        patch_index_zooms: Dict[int, torch.Tensor] = {}
        for zoom in self.sampling_zooms.keys():
            patch_index_zooms[zoom] = torch.tensor(self.index_map[zoom][index][-1])

        source_zooms_groups_out: List[Any] = []
        target_zooms_groups_out: List[Any] = []
        mask_zooms_groups: List[Any] = []
        emb_groups: List[Dict[str, Any]] = []

        emb: Dict[str, Any] = {}
        StaticVariableEmbedder: Optional[Dict[int, torch.Tensor]] = None
        # emb['DensityEmbedder'] = torch.tensor([selected_var_ids[group] for group in group_keys])
        for group_idx, group in enumerate(group_keys):
            if group == 'embedding': 
                # Extract static embeddings once so they can be attached to other groups.
                StaticVariableEmbedder = source_zooms_groups[group_idx]
                StaticVariableEmbedder = dict(zip(StaticVariableEmbedder.keys(), 
                                                  [rearrange(t, 'v (b t) n f d-> b t n (v f d)', b=self.load_n_samples_time) for t in StaticVariableEmbedder.values()]))

        for group_idx, group in enumerate(group_keys):
            if group != 'embedding': 
                source_zooms: Any
                target_zooms: Any
                mask_group: Dict[int, torch.Tensor]
                source_zooms, target_zooms, _, mask_group = self._finalize_group(
                    source_zooms_groups[group_idx],
                    target_zooms_groups[group_idx],
                    time_zooms,
                    mask_mapping_zooms_groups[group_idx],
                    patch_index_zooms,
                    hr_dopout
                )
                source_zooms_groups_out.append(source_zooms)
                target_zooms_groups_out.append(target_zooms)
                mask_zooms_groups.append(mask_group)

                emb_group: Dict[str, Any] = emb.copy()
                emb_group['VariableEmbedder'] = torch.tensor(list(var_indices[group])).view(-1)
                emb_group['MGEmbedder'] = emb_group['VariableEmbedder']

                if StaticVariableEmbedder is not None:
                    emb_group['StaticVariableEmbedder'] = StaticVariableEmbedder

                for zoom, time_zoom in time_zooms.items():
                    # Keep time embeddings aligned with the batch/time stacking strategy.
                    time_zooms[zoom] = time_zoom.reshape(self.load_n_samples_time, -1)
                    
                emb_group['TimeEmbedder'] = time_zooms
                emb_groups.append(emb_group)
        
            source_zooms_groups_out_: Dict[int, torch.Tensor] = {}
            target_zooms_groups_out_: Dict[int, torch.Tensor] = {}
            mask_zooms_groups_: Dict[int, torch.Tensor] = {}
            if self.variables_as_features:
                for zoom in source_zooms_groups_out[0].keys():
                    source_zooms_groups_out_[zoom] = torch.concat([group[zoom] for group in source_zooms_groups_out],dim=-1)
                    target_zooms_groups_out_[zoom] =  torch.concat([group[zoom] for group in target_zooms_groups_out],dim=-1)
                    mask_zooms_groups_[zoom] = torch.concat([group[zoom] for group in mask_zooms_groups], dim=-1)

                emb = {'StaticVariableEmbedder': emb_groups[0]['StaticVariableEmbedder'],
                       'TimeEmbedder': emb_groups[0]['TimeEmbedder'],
                       'VarialeEmbedder': torch.zeros(source_zooms_groups_out_[zoom].shape[-1], dtype=torch.long)}

                emb_groups = [emb]
                source_zooms_groups_out = [source_zooms_groups_out_]
                target_zooms_groups_out = [target_zooms_groups_out_]
                mask_zooms_groups = [mask_zooms_groups_]

        return source_zooms_groups_out, target_zooms_groups_out, mask_zooms_groups, emb_groups, patch_index_zooms


    def __len__(self) -> int:
        """
        Return the length of the dataset.

        :return: Number of available samples.
        """
        return self.len_dataset
