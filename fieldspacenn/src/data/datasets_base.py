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
warnings.filterwarnings("ignore", message="ZarrUserWarning.*")

from ..modules.grids.grid_utils import get_coords_as_tensor,get_grid_type_from_var,get_mapping_weights,to_zoom, encode_zooms, decode_zooms
from . import normalizer as normalizers

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
    uniform_random = torch.rand(size)
    skewed_random = max_p * (1 - uniform_random ** exponent)
    return skewed_random

def invert_dict(d: Mapping[Any, Any]) -> Dict[Any, List[Any]]:
    """
    Invert a dictionary mapping values to lists of keys.

    :param d: Input mapping to invert.
    :return: Dictionary that groups original keys by their values.
    """
    inverted_d = {}
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

        # shift target timesteps per zoom; default to no shift
        self.shift_n_ts_target: Dict[int, int] = dict(
            zip(
                self.sampling_zooms.keys(),
                [int(v.get("shift_n_ts_target", 0)) for v in self.sampling_zooms.values()],
            )
        )

        self.mask_ts_mode: str = mask_ts_mode
        self.p_dropout_all_zooms: Dict[int, float] = dict(
            zip(self.sampling_zooms.keys(), [v.get("p_drop", 0) for v in self.sampling_zooms.values()])
        )
        self.mask_n_last_ts_zooms: Dict[int, int] = dict(
            zip(self.sampling_zooms.keys(), [v.get("mask_n_last_ts", 0) for v in self.sampling_zooms.values()])
        )

        self.p_dropout_all: float = p_dropout_all


        if "files" in self.data_dict['source'].keys():
            all_files = self.data_dict['source']["files"]

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
        unique_time_steps_past = len(torch.tensor(self.zoom_time_steps_past).unique()) == 1
        unique_time_steps_future = len(torch.tensor(self.zoom_time_steps_future).unique()) == 1
        unique_zoom_patch_sample = len(torch.tensor(self.zoom_patch_sample).unique()) == 1

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
            ds = xr.open_dataset(file)
            self.time_steps_files.append(len(ds.time))

        # Build index map of (file, time window, region) per zoom.
        self.index_map: Dict[int, List[Tuple[int, List[int], int]]] = dict(
            zip(self.zooms, [[] for _ in self.zooms])
        )
        for file_idx, total_timesteps in enumerate(self.time_steps_files):
            # ensure both source window and shifted target window are inside the dataset
            start_bounds = []
            end_bounds = []
            for zoom in self.zooms:
                shift_target = self.shift_n_ts_target.get(zoom, 0)
                n_past_ts = self.sampling_zooms[zoom]['n_past_ts']
                n_future_ts = self.sampling_zooms[zoom]['n_future_ts']

                start_bounds.append(n_past_ts)
                end_bounds.append(total_timesteps - 1 - n_future_ts - shift_target)

            start_idx = max(start_bounds)
            end_idx = min(end_bounds)

            if end_idx < start_idx:
                continue

            if self.sample_timesteps is None:
                time_indices = list(range(start_idx, end_idx + 1))
            else:
                time_indices = [t for t in self.sample_timesteps if start_idx <= t <= end_idx]

            # drop incomplete groups instead of raising when load_n_samples_time does not divide cleanly
            n_complete = len(time_indices) // self.load_n_samples_time
            time_indices = time_indices[: n_complete * self.load_n_samples_time]

            if len(time_indices) == 0:
                continue

            time_entries = np.array(time_indices).reshape(-1, self.load_n_samples_time)

            for time_entry in time_entries:
                for zoom in self.zooms:
                    for region_idx_max in range(self.indices[max(self.zooms)].shape[0]):
                        if self.sampling_zooms[zoom]['zoom_patch_sample'] == -1:
                            region_idx_zoom = 0
                        else:
                            region_idx_zoom = region_idx_max//4**(self.sampling_zooms[max(self.zooms)]['zoom_patch_sample'] - self.sampling_zooms[zoom]['zoom_patch_sample'])

                        self.index_map[zoom].append((file_idx, list(time_entry), region_idx_zoom))
                
        #self.index_map = {z: np.array(idx_map) for z, idx_map in self.index_map.items()}

        # Build variable group indices for embedding and masking.
        all_variables = []
        variable_ids = {}
        all_ids = []
        self.group_ids: Dict[str, int] = {}
        offset = 0
        for group_id, (group, vars) in enumerate(self.data_dict['variables'].items()):
            all_variables += vars
            variable_ids[group] = np.arange(len(vars)) + offset
            all_ids = all_ids+list(variable_ids[group])
            offset = len(variable_ids[group])
            self.group_ids[group] = (group_id)
        
        self.all_variable_ids: Dict[str, int] = dict(zip(all_variables, all_ids))

        grid_types = [get_grid_type_from_var(ds, var) for var in all_variables]
        self.vars_grid_types: Dict[str, Any] = dict(zip(all_variables, grid_types))
        self.grid_types: np.ndarray = np.unique(grid_types)

        self.grid_types_vars: Dict[Any, List[str]] = invert_dict(self.vars_grid_types)
        for var, gtype in zip(all_variables, grid_types):
            self.grid_types_vars[gtype].append(var)

        unique_files = np.unique(np.array(all_files))

        self.single_source: bool = len(unique_files) == 1
        self.mapping: Dict[int, Dict[Any, Any]] = {}
        if self.single_source:
            # Single-source: build a shared mapping at the highest zoom and reuse across zooms.
            coords = [
                get_coords_as_tensor(ds, grid_type=grid_type) for grid_type in self.grid_types
            ]
            mapping_hr = dict(
                zip(self.grid_types, [mapping_fcn(coords_, max(self.zooms))[max(self.zooms)] for coords_ in coords])
            )
            self.mapping[max(self.zooms)] = mapping_hr
        else:
            for zoom in self.zooms:
                # Multi-source: build a per-zoom mapping using that zoom's grid.
                mapping_grid_type = {}
                for grid_type in self.grid_types:
                    ds = xr.open_dataset(self.data_dict['source'][zoom]['files'][0])
                    coords = get_coords_as_tensor(ds, grid_type=grid_type)
                    mapping_grid_type[grid_type] = mapping_fcn(coords, zoom)[zoom]
                self.mapping[zoom] = mapping_grid_type

        self.load_once: bool = (
            unique_time_steps_past and unique_time_steps_future and unique_zoom_patch_sample and self.single_source
        )

        with open(norm_dict) as json_file:
            norm_dict = json.load(json_file)

        self.var_normalizers: Dict[int, Dict[str, Any]] = {}
        for zoom in self.zooms:
            self.var_normalizers[zoom] = {}
            for var in all_variables:
                if str(zoom) in norm_dict[var].keys():
                    # Zoom-specific stats override global stats when available.
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
            ds_source = xr.open_dataset(file_path_source, decode_times=False)
        else:
            ds_source = xr.load_dataset(file_path_source, decode_times=False)

        if file_path_target is None:
            ds_target = None

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
        mapping: Mapping[Any, Any],
        mapping_zoom: int,
        zoom: int,
    ) -> xr.Dataset:
        """
        Slice a dataset to the time window and spatial patch for a zoom.

        :param ds: Input xarray dataset to slice.
        :param time_indices: Center time indices for the sample window.
        :param patch_idx: Patch index within the zoom grid.
        :param mapping: Grid mapping dictionary keyed by grid type.
        :param mapping_zoom: Zoom level associated with the mapping.
        :param zoom: Zoom level of the requested data.
        :return: Sliced dataset for the requested time window and patch.
        """
        # Fetch raw patch indices
        isel_dict = {"time": time_indices}
        patch_dim = [d for d in ds.dims if "cell" in d or "ncells" in d]
        patch_dim = patch_dim[0] if patch_dim else None

        for grid_type, variables_grid_type in self.grid_types_vars.items():
            mapping = mapping[grid_type]
            patch_indices = self.get_indices_from_patch_idx(zoom, patch_idx)

            # Resolve indices either on the target grid (post-map) or the source grid (pre-map).
            post_map = mapping_zoom > zoom
            if post_map:
                indices = mapping['indices'][..., [0]].reshape(-1, 4 ** (mapping_zoom - zoom))
                if patch_dim:
                    isel_dict[patch_dim] = indices.view(-1)

            else:
                indices = mapping['indices'][..., [0]]

                if patch_dim:
                    isel_dict[patch_dim] = indices[patch_indices].view(-1)

        ds_zoom = ds.isel(isel_dict)
    
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
            shape ``(v, t, n, d, f)`` (matching the ``(b, v, t, n, d, f)`` base shape with
            ``b`` handled by the caller), ``data_time`` is a tensor of shape ``(t,)``,
            and ``drop_mask`` is a tensor of shape ``(v, t, n, d, 1)``.
        """
        # Fetch raw patch indices
        drop_mask_ = drop_mask.clone() if drop_mask is not None else None
        if drop_mask_ is not None and drop_mask_.ndim == 2:
            drop_mask_ = drop_mask_.unsqueeze(0)

        patch_dim_candidates = [d for d in ds.dims if "cell" in d or "ncells" in d]
        patch_dim = patch_dim_candidates[0] if patch_dim_candidates else None

        data_g = []
        for grid_type, variables_grid_type in self.grid_types_vars.items():
            variables = [var for var in variables_sample if var in variables_grid_type]
            if not variables:
                continue

            mapping = mapping[grid_type]

            patch_indices = self.get_indices_from_patch_idx(zoom, patch_idx)

            mask = get_mapping_weights(mapping)[..., 0].view(1, 1, -1, 1, 1)

            # Map indices differently depending on whether we are projecting from a higher zoom.
            post_map = mapping_zoom > zoom
            if post_map:
                indices = mapping['indices'][..., [0]].reshape(-1, 4 ** (mapping_zoom - zoom))

            else:
                indices = mapping['indices'][..., [0]]
                mask = mask[:, :, patch_indices]

                if drop_mask_ is not None:
                    drop_mask_ = drop_mask_[:, :, patch_indices]

            ds_variables = ds[variables]
            data_g = torch.from_numpy(ds_variables.to_array().to_numpy()).unsqueeze(dim=-1)

            if self.normalize_data:
                for k, variable in enumerate(variables):
                    data_g[k] = self.var_normalizers[zoom][variable].normalize(data_g[k])

            if 'level' not in ds_variables.dims:
                data_g = data_g.unsqueeze(dim=2)

            data_g = data_g.transpose(2,3)

            if not patch_dim and post_map:
                data_g = data_g.reshape(data_g.shape[0], data_g.shape[1], -1)[:, :, indices.view(-1)]

        if drop_mask_ is not None and mask.dtype!=torch.bool:
            mask = (1-1.*drop_mask_.unsqueeze(dim=-1)) * mask
            mask = mask.expand_as(data_g)

        elif drop_mask_ is not None:
            mask = torch.logical_or(drop_mask, mask.expand_as(drop_mask))

        else:
            mask = None
        
        if mask is not None and not (mask == False).any():
            mask = mask.expand_as(data_g)

        data_g, mask = to_zoom(data_g, mapping_zoom, zoom, mask=mask, binarize_mask=self.output_binary_mask)

        if post_map:
            data_g = data_g[:, :, patch_indices]
            mask = mask[:, :, patch_indices] if mask is not None else None

        if mask is not None:
            mask = torch.logical_not(mask[...,[0]]) if mask.dtype==torch.bool else mask[...,[0]]

        return data_g, mask

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
        :return: Boolean mask tensor of shape ``(v, t, n)`` aligned to the
            ``(b, v, t, n, d, f)`` base shape (with ``b, d, f`` handled elsewhere).
        """
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
            drop_mask[:, drop_mask_p] = True

        elif p_dropout > 0 and drop_groups:
            drop_mask_p = (torch.rand((ng, nt, n)) < p_dropout.view(-1, 1, 1)).bool()
            drop_mask[drop_mask_p] = True

        elif p_dropout > 0 and drop_timesteps:
            drop_mask_p = (torch.rand(nt) < p_dropout).bool()
            drop_mask[:, drop_mask_p] = True

        if n_drop_groups != -1 and n_drop_groups < ng:
            not_drop_vars = torch.randperm(ng)[:(ng - n_drop_groups)]
            drop_mask[not_drop_vars] = (drop_mask[not_drop_vars] * 0).bool()

        return drop_mask
  
    def _finalize_group(
        self,
        data_source: Mapping[int, torch.Tensor],
        data_target: Mapping[int, torch.Tensor],
        mask_mapping_zooms: Mapping[int, torch.Tensor],
        patch_index_zooms: Mapping[int, torch.Tensor],
        hr_dopout: bool,
    ) -> Tuple[Any, Any, Dict[int, Dict[str, Any]], Dict[int, torch.Tensor]]:
        """
        Finalize group data by applying masks, reshaping, and zoom transforms.

        :param data_source: Mapping from zoom to source tensor of shape ``(v, t, n, d, f)``.
        :param data_target: Mapping from zoom to target tensor of shape ``(v, t, n, d, f)``.
        :param mask_mapping_zooms: Mapping from zoom to mask tensor of shape ``(v, t, n, d, f)``.
        :param patch_index_zooms: Mapping from zoom to patch index tensor of shape ``(1,)``.
        :param hr_dopout: Whether high-resolution dropout is active.
        :return: Tuple ``(data_source, data_target, sample_configs, mask_mapping_zooms)`` where
            data tensors are reshaped to ``(b, v, t, n, d, f)`` (or ``(b, 1, t, n, 1, f)``
            when ``variables_as_features`` is enabled), aligning with the base
            ``(b, v, t, n, d, f)`` convention.
        """
        sample_configs = copy.deepcopy(self.sampling_zooms)
        if not data_source:
            for key, value in patch_index_zooms.items():
                if key in sample_configs:
                    sample_configs[key]['patch_index'] = value
            return {}, {}, {}, sample_configs
        if data_target is None:
            # Defer target construction until here to avoid masking it with source dropouts.
            data_target = {zoom: data_source[zoom].clone() for zoom in data_source.keys()}
        else:
            for zoom in list(data_source.keys()):
                if zoom not in data_target or data_target[zoom] is None:
                    data_target[zoom] = data_source[zoom].clone()

        if not hr_dopout and self.p_dropout_all > 0:
            drop = False
            for zoom in sorted(self.sampling_zooms.keys()):

                if self.p_dropout_all_zooms[zoom] > 0 and not drop:
                    drop = torch.rand(1) < self.p_dropout_all_zooms[zoom]

                if drop:
                    mask_mapping_zooms[zoom] = torch.ones_like(data_source[zoom], dtype=bool)

        # Apply computed masks to zero-out dropped entries.
        for zoom in data_source.keys():
            # mask data
            if mask_mapping_zooms[zoom] is not None:
                if mask_mapping_zooms[zoom].dtype == torch.float:
                    mask_zoom = mask_mapping_zooms[zoom] == 0
                else:
                    mask_zoom = mask_mapping_zooms[zoom]
            
                if mask_zoom.any():
                    data_source[zoom][mask_zoom.expand_as(data_source[zoom])] = 0

            if self.variables_as_features:
                data_source[zoom] = rearrange(data_source[zoom], 'v (b t) n d f -> b 1 t n 1 (v d f)', b=self.load_n_samples_time)
                data_target[zoom] = rearrange(data_target[zoom], 'v (b t) n d f -> b 1 t n 1 (v d f)', b=self.load_n_samples_time)

                if mask_mapping_zooms[zoom] is None:
                    mask_mapping_zooms[zoom] = torch.empty((self.load_n_samples_time))
                else:
                    mask_mapping_zooms[zoom] = rearrange(mask_mapping_zooms[zoom], 'v (b t) n d f -> b 1 t n 1 (v d f)', b=self.load_n_samples_time)
            else:
                data_source[zoom] = rearrange(data_source[zoom], 'v (b t) n d f -> b v t n d f', b=self.load_n_samples_time)
                data_target[zoom] = rearrange(data_target[zoom], 'v (b t) n d f -> b v t n d f', b=self.load_n_samples_time)

                if mask_mapping_zooms[zoom] is None:
                    mask_mapping_zooms[zoom] = torch.empty((self.load_n_samples_time))
                else:
                    mask_mapping_zooms[zoom] = rearrange(mask_mapping_zooms[zoom], 'v (b t) n d f -> b v t n d f', b=self.load_n_samples_time)

        for key, value in patch_index_zooms.items():
            if key in sample_configs:
                sample_configs[key]['patch_index'] = value
        
        if self.output_max_zoom_only:
            max_zoom = max(data_source.keys())
            data_source = data_source[max_zoom]
            data_target = data_target[max_zoom]
        else:
            data_source = encode_zooms(data_source, sample_configs, patch_index_zooms)
            data_target = encode_zooms(data_target, sample_configs, patch_index_zooms)

        # Optionally mask the last timesteps and repeat or zero them out.
        if any([z.get('mask_n_last_ts', 0) > 0 for z in self.sampling_zooms.values()]):
            for zoom, sampling_zoom in self.sampling_zooms.items():
                mask_n_last_ts = sampling_zoom.get('mask_n_last_ts', 0)
                if mask_n_last_ts > 0:
                    time_len = data_source[zoom].shape[2]
                    n_mask = min(mask_n_last_ts, time_len)
                    if n_mask == 0:
                        continue
                    
                    if mask_mapping_zooms[zoom].numel()==1:
                        mask_mapping_zooms[zoom] = torch.zeros_like(data_source[zoom],dtype=bool)

                    mask_mapping_zooms[zoom][:, :, -n_mask:] = True 
                   
                    if self.mask_ts_mode == 'repeat' and time_len > n_mask:
                        repeat_source = data_source[zoom][:, :, -(n_mask + 1)].unsqueeze(2)
                        data_source[zoom][:, :, -n_mask:] = repeat_source.expand_as(data_source[zoom][:, :, -n_mask:])
                    else:
                        data_source[zoom][:, :, -n_mask:] = 0.

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
        selected_vars = {}
  
        var_indices = {}
        group_keys = list(self.data_dict['variables'].keys())
        # Sample variables per group to build a compact input for this item.
        for group in group_keys:
            variables = self.data_dict['variables'][group]
            sample_size = len(variables) if self.n_sample_variables == -1 else min(self.n_sample_variables, len(variables))
            var_indices[group] = np.arange(len(variables))

            if sample_size != len(variables):
                var_indices[group] = np.random.choice(var_indices[group], sample_size, replace=False)

            selected_vars[group] = np.array(variables)[var_indices[group]]
            

        hr_dopout = self.p_dropout > 0 and torch.rand(1) > (self.p_dropout_all)

        # Only build a global dropout mask when a single source ensures shared indexing.
        if self.single_source and hr_dopout:
            nt = 1 + self.max_time_step_future + self.max_time_step_past
            total_vars = sum(len(selected_vars[group]) for group in selected_vars.keys())
            drop_mask_input = self.get_mask(
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

       
        source_zooms_groups = [{} for _ in group_keys]
        target_zooms_groups = [{} for _ in group_keys]
        data_time_zooms = {}
        mask_mapping_zooms_groups = [{} for _ in group_keys]
        patch_index_zooms = {}

        loaded = False
        for zoom in self.zooms:
            file_index, time_indices, patch_index = self.index_map[zoom][index]
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

            # Align the global dropout mask to this zoom's time window.
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
                for indices in var_indices.values():
                    drop_mask_zoom_groups.append(drop_mask_zoom[indices].unsqueeze(0))
    
            start_times = np.array(time_indices) - self.sampling_zooms[zoom]['n_past_ts'] 
            end_times = np.array(time_indices) + self.sampling_zooms[zoom]['n_future_ts']

            time_indices = np.stack(
                [np.arange(s, e + 1) for s, e in zip(start_times, end_times)],
                axis=0
            ).reshape(-1)

            ds_source_zoom = self.select_ranges(ds_source,
                    time_indices,
                    patch_index,
                    self.mapping[mapping_zoom],
                    mapping_zoom,
                    zoom)
            
            data_time_zooms[zoom] = torch.from_numpy(ds_source_zoom.time.values).view(self.load_n_samples_time,-1)
            
            if ds_target is not None or self.shift_n_ts_target.get(zoom,0)>0:
                ds_target = ds_source if ds_target is not None else ds_target
                ds_target_zoom = self.select_ranges(
                    ds_target,
                    time_indices + self.shift_n_ts_target.get(zoom,0),
                    patch_index,
                    self.mapping[mapping_zoom],
                    mapping_zoom,
                    zoom,
                )
            else:
                ds_target_zoom = None

            for group_idx, group in enumerate(group_keys):
                data_source, drop_mask_zoom_group = self.get_data(
                    ds_source_zoom,
                    patch_index,
                    selected_vars[group],
                    self.mapping[mapping_zoom],
                    mapping_zoom,
                    zoom,
                    drop_mask=drop_mask_zoom_groups[group_idx],
                )

                if ds_target is not None:
                    data_target,  _ = self.get_data(
                        ds_target_zoom,
                        patch_index,
                        selected_vars[group],
                        self.mapping[mapping_zoom],
                        mapping_zoom,
                        zoom,
                    )
                else:
                    data_target = None

                source_zooms_groups[group_idx][zoom] = data_source
                target_zooms_groups[group_idx][zoom] = data_target
                mask_mapping_zooms_groups[group_idx][zoom] = drop_mask_zoom_group

            patch_index_zooms[zoom] = torch.tensor(self.index_map[zoom][index][-1])
            
        ds_source_zoom.close()

        if ds_target_zoom is not None:
            ds_target_zoom.close()

        source_zooms_groups_out = []
        target_zooms_groups_out = []
        mask_zooms_groups = []
        emb_groups = []

        emb = {}
        StaticVariableEmbedder = None
        # emb['DensityEmbedder'] = torch.tensor([selected_var_ids[group] for group in group_keys])
        for group_idx, group in enumerate(group_keys):
            if group == 'embedding': 
                # Extract static embeddings once so they can be attached to other groups.
                StaticVariableEmbedder = source_zooms_groups[group_idx]
                StaticVariableEmbedder = dict(zip(StaticVariableEmbedder.keys(), 
                                                  [rearrange(t, 'v (b t) n f d-> b t n (v f d)', b=self.load_n_samples_time) for t in StaticVariableEmbedder.values()]))

        for group_idx, group in enumerate(group_keys):
            if group != 'embedding': 
                source_zooms, target_zooms, _, mask_group = self._finalize_group(
                    source_zooms_groups[group_idx],
                    target_zooms_groups[group_idx],
                    mask_mapping_zooms_groups[group_idx],
                    patch_index_zooms,
                    hr_dopout
                )
                source_zooms_groups_out.append(source_zooms)
                target_zooms_groups_out.append(target_zooms)
                mask_zooms_groups.append(mask_group)

                emb_group = emb.copy()
                emb_group['VariableEmbedder'] = torch.tensor(list(var_indices[group])).view(1,-1).repeat_interleave(self.load_n_samples_time,dim=0)
                emb_group['MGEmbedder'] = emb_group['VariableEmbedder']

                if StaticVariableEmbedder is not None:
                    emb_group['StaticVariableEmbedder'] = StaticVariableEmbedder

                    
                emb_group['TimeEmbedder'] = data_time_zooms
                emb_groups.append(emb_group)
        
            source_zooms_groups_out_ = {}
            target_zooms_groups_out_ = {}
            mask_zooms_groups_ = {}

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
        
        for zoom, indices in patch_index_zooms.items():
            patch_index_zooms[zoom] = indices.view(1).repeat_interleave(self.load_n_samples_time, dim=0)

        return source_zooms_groups_out, target_zooms_groups_out, mask_zooms_groups, emb_groups, patch_index_zooms


    def __len__(self) -> int:
        """
        Return the length of the dataset.

        :return: Number of available samples.
        """
        return self.len_dataset
