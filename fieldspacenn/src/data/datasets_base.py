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

from ..modules.grids.grid_utils import get_coords_as_tensor,get_grid_type_from_var,get_mapping_weights,to_zoom, encode_zooms, decode_zooms, get_zoom_from_npix
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
    @staticmethod
    def _normalize_groups(groups: Mapping[str, Sequence[str]], context: str) -> Dict[str, List[str]]:
        """
        Normalize a group mapping to ``Dict[str, List[str]]``.

        :param groups: Mapping from group name to variable sequence.
        :param context: Error-context string used in validation messages.
        :return: Normalized group mapping.
        """
        normalized: Dict[str, List[str]] = {}
        for group_name, variables in groups.items():
            if isinstance(variables, str):
                variables_list = [variables]
            elif isinstance(variables, (list, tuple, ListConfig)):
                variables_list = list(variables)
            else:
                raise TypeError(
                    f"{context}: group '{group_name}' must map to a list of variables, "
                    f"got {type(variables)}."
                )

            if len(variables_list) == 0:
                raise ValueError(f"{context}: group '{group_name}' must contain at least one variable.")

            normalized[str(group_name)] = [str(v) for v in variables_list]

        if len(normalized) == 0:
            raise ValueError(f"{context}: expected at least one variable group.")
        return normalized

    @classmethod
    def _resolve_groups_from_sampling(
        cls,
        sampling_zooms: Mapping[int, Mapping[str, Any]],
        default_groups: Optional[Mapping[str, Sequence[str]]],
        side: str,
    ) -> Dict[int, Dict[str, List[str]]]:
        """
        Resolve per-zoom variable groups from sampling configuration.

        :param sampling_zooms: Sampling dictionary keyed by zoom.
        :param default_groups: Optional fallback variable groups.
        :param side: Side label used in errors (``source`` or ``target``).
        :return: Mapping of zoom to normalized group mapping.
        """
        resolved: Dict[int, Dict[str, List[str]]] = {}
        default_groups_norm = None
        if default_groups:
            default_groups_norm = cls._normalize_groups(
                default_groups, context=f"default `{side}` groups"
            )

        for zoom, sampling in sampling_zooms.items():
            zoom_groups_raw = sampling.get("groups")
            if zoom_groups_raw is None:
                if default_groups_norm is None:
                    raise ValueError(
                        f"Missing `groups` in sampling config for {side} zoom {zoom}. "
                        f"Provide `{side}` variables either in `sampling_zooms*.{zoom}.groups` "
                        "or via dataset-level `variables`."
                    )
                zoom_groups = copy.deepcopy(default_groups_norm)
            else:
                zoom_groups = cls._normalize_groups(
                    zoom_groups_raw, context=f"{side} zoom {zoom} groups"
                )
            resolved[int(zoom)] = zoom_groups

        return resolved

    @staticmethod
    def _validate_consistent_groups_across_zooms(
        groups_by_zoom: Mapping[int, Mapping[str, Sequence[str]]],
        side: str,
    ) -> Dict[str, List[str]]:
        """
        Validate that variable groups are consistent across zooms for one side.

        :param groups_by_zoom: Per-zoom group definitions.
        :param side: Side label used in errors.
        :return: Canonical group definition for the side.
        """
        canonical: Optional[Dict[str, List[str]]] = None
        ref_zoom: Optional[int] = None
        for zoom in sorted(groups_by_zoom.keys()):
            groups_zoom = {
                str(group): [str(v) for v in variables]
                for group, variables in groups_by_zoom[zoom].items()
            }
            if canonical is None:
                canonical = groups_zoom
                ref_zoom = zoom
                continue
            if groups_zoom != canonical:
                raise ValueError(
                    f"Inconsistent `{side}` groups across zooms. "
                    f"zoom {ref_zoom}: {canonical}, zoom {zoom}: {groups_zoom}."
                )
        if canonical is None:
            raise ValueError(f"No `{side}` groups resolved from sampling configuration.")
        return canonical

    @staticmethod
    def _normalize_aliases(
        aliases: Optional[Mapping[str, Union[str, Sequence[str]]]],
        context: str,
    ) -> Dict[str, List[str]]:
        """
        Normalize variable aliases to ``Dict[canonical_name, List[alias_name]]``.

        :param aliases: Optional mapping from canonical variable name to one or more aliases.
        :param context: Error-context string used in validation messages.
        :return: Normalized alias mapping.
        """
        if aliases is None:
            return {}

        normalized: Dict[str, List[str]] = {}
        for canonical_var, alias_values in aliases.items():
            canonical_name = str(canonical_var)
            if isinstance(alias_values, str):
                alias_list = [alias_values]
            elif isinstance(alias_values, (list, tuple, ListConfig)):
                alias_list = list(alias_values)
            else:
                raise TypeError(
                    f"{context}: aliases for '{canonical_name}' must be a string or list of strings, "
                    f"got {type(alias_values)}."
                )

            # Keep canonical first so it has priority when present in a file.
            candidate_names = [canonical_name] + [str(name) for name in alias_list]
            deduped_names = list(dict.fromkeys(candidate_names))
            normalized[canonical_name] = deduped_names

        return normalized

    def _get_alias_candidates(self, variable: str, side: str) -> List[str]:
        """
        Build candidate dataset variable names for a canonical variable.

        :param variable: Canonical variable name used in dataset groups.
        :param side: Dataset side (``source`` or ``target``).
        :return: Ordered candidate names with duplicates removed.
        """
        variable = str(variable)
        if side not in {"source", "target"}:
            raise ValueError(f"Unsupported side '{side}'. Expected 'source' or 'target'.")

        candidates: List[str] = [variable]
        side_aliases = self.variable_aliases_source if side == "source" else self.variable_aliases_target
        for alias_map in (side_aliases, self.variable_aliases):
            if variable in alias_map:
                candidates.extend(alias_map[variable])
        return list(dict.fromkeys(str(name) for name in candidates))

    def _resolve_variable_name_in_dataset(
        self,
        ds: xr.Dataset,
        variable: str,
        side: str,
        file_path: str,
    ) -> str:
        """
        Resolve a canonical variable name to the concrete name present in one dataset.

        :param ds: Input xarray dataset.
        :param variable: Canonical variable name.
        :param side: Dataset side (``source`` or ``target``).
        :param file_path: Dataset file path used for diagnostics and cache keying.
        :return: Dataset variable name available in ``ds.data_vars``.
        """
        candidates = self._get_alias_candidates(variable, side)
        available_vars = set(ds.data_vars.keys())
        matches = [name for name in candidates if name in available_vars]

        if len(matches) == 0:
            preview = list(ds.data_vars.keys())
            raise KeyError(
                f"Could not resolve variable '{variable}' for {side} file '{file_path}'. "
                f"Tried aliases {candidates}. "
                f"Available vars: {preview[:10]}{'...' if len(preview) > 10 else ''}."
            )

        if variable in matches:
            return variable

        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous aliases for variable '{variable}' in {side} file '{file_path}'. "
                f"Matched candidates: {matches}. Keep only one alias present in a file."
            )

        return matches[0]

    def _get_resolved_variable_map(
        self,
        ds: xr.Dataset,
        variables: Sequence[str],
        side: str,
        file_path: str,
    ) -> Dict[str, str]:
        """
        Resolve canonical variable names to concrete dataset names with per-file caching.

        :param ds: Input xarray dataset.
        :param variables: Canonical variable names to resolve.
        :param side: Dataset side (``source`` or ``target``).
        :param file_path: Dataset file path used for diagnostics and cache keying.
        :return: Mapping ``canonical_name -> dataset_name``.
        """
        file_key = str(file_path)
        cache_key = (side, file_key)
        cache = self._resolved_variable_names_cache.setdefault(cache_key, {})

        variables_unique = list(dict.fromkeys(str(var) for var in variables))
        for variable in variables_unique:
            if variable not in cache:
                cache[variable] = self._resolve_variable_name_in_dataset(
                    ds=ds,
                    variable=variable,
                    side=side,
                    file_path=file_key,
                )

        resolved = {variable: cache[variable] for variable in variables_unique}
        reverse_map: Dict[str, List[str]] = {}
        for canonical_name, dataset_name in resolved.items():
            reverse_map.setdefault(dataset_name, []).append(canonical_name)
        collisions = {
            dataset_name: canonical_names
            for dataset_name, canonical_names in reverse_map.items()
            if len(canonical_names) > 1
        }
        if collisions:
            raise ValueError(
                f"Alias collision in {side} file '{file_key}': {collisions}. "
                "Each canonical variable in a batch must resolve to a unique dataset variable."
            )

        return resolved

    def _resolve_norm_key(self, variable: str, norm_dict: Mapping[str, Any]) -> str:
        """
        Resolve which entry in ``norm_dict`` should be used for a canonical variable.

        :param variable: Canonical variable name.
        :param norm_dict: Parsed normalization dictionary.
        :return: Key in ``norm_dict`` to use for the variable.
        """
        if variable in norm_dict:
            return variable

        candidates = self._get_alias_candidates(variable, "source") + self._get_alias_candidates(
            variable, "target"
        )
        candidates = list(dict.fromkeys(candidates))
        matches = [name for name in candidates if name in norm_dict]

        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"Multiple normalization entries match variable '{variable}': {matches}. "
                "Keep exactly one matching key in norm_dict."
            )

        available = list(norm_dict.keys())
        raise KeyError(
            f"Missing normalization stats for variable '{variable}'. "
            f"Tried keys {candidates}. "
            f"Available keys: {available[:10]}{'...' if len(available) > 10 else ''}."
        )

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
        variables: Optional[Mapping[str, Sequence[str]]] = None,
        variable_aliases: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
        variable_aliases_source: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
        variable_aliases_target: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
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
        :param variables: Optional default variable groups used when a zoom config
            does not define ``groups``.
        :param variable_aliases: Optional aliases shared by source and target sides.
            Format: ``{canonical_name: [alias_name_1, alias_name_2, ...]}``.
        :param variable_aliases_source: Optional source-side aliases that override or
            extend ``variable_aliases``.
        :param variable_aliases_target: Optional target-side aliases that override or
            extend ``variable_aliases``.
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
        self.variable_aliases: Dict[str, List[str]] = self._normalize_aliases(
            variable_aliases,
            context="`variable_aliases`",
        )
        self.variable_aliases_source: Dict[str, List[str]] = self._normalize_aliases(
            variable_aliases_source,
            context="`variable_aliases_source`",
        )
        self.variable_aliases_target: Dict[str, List[str]] = self._normalize_aliases(
            variable_aliases_target,
            context="`variable_aliases_target`",
        )
        self._resolved_variable_names_cache: Dict[Tuple[str, str], Dict[str, str]] = {}

        self.load_n_samples_time: int = load_n_samples_time

        if not hasattr(self, "sampling_zooms_source") or self.sampling_zooms_source is None:
            self.sampling_zooms_source = copy.deepcopy(self.sampling_zooms)
        if not hasattr(self, "sampling_zooms_target") or self.sampling_zooms_target is None:
            self.sampling_zooms_target = copy.deepcopy(self.sampling_zooms)

        self.sampling_zooms_source = {
            int(z): copy.deepcopy(v) for z, v in self.sampling_zooms_source.items()
        }
        self.sampling_zooms_target = {
            int(z): copy.deepcopy(v) for z, v in self.sampling_zooms_target.items()
        }
        self.sampling_zooms = {int(z): copy.deepcopy(v) for z, v in self.sampling_zooms.items()}
        for zoom, sampling in self.sampling_zooms_target.items():
            if zoom not in self.sampling_zooms:
                self.sampling_zooms[zoom] = copy.deepcopy(sampling)

        self.zooms_source: List[int] = sorted(list(self.sampling_zooms_source.keys()))
        self.zooms_target: List[int] = sorted(list(self.sampling_zooms_target.keys()))
        self.zooms: List[int] = sorted(list(self.sampling_zooms.keys()))
        self.max_zoom_source: int = max(self.zooms_source)
        self.max_zoom_target: int = max(self.zooms_target)

        self.mask_ts_mode: str = mask_ts_mode
        self.p_dropout_all_zooms: Dict[int, float] = dict(
            zip(
                self.sampling_zooms_source.keys(),
                [v.get("p_drop", 0) for v in self.sampling_zooms_source.values()],
            )
        )
        self.mask_n_last_ts_zooms: Dict[int, int] = dict(
            zip(
                self.sampling_zooms_source.keys(),
                [v.get("mask_n_last_ts", 0) for v in self.sampling_zooms_source.values()],
            )
        )
        # shift target timesteps per zoom; default to no shift
        self.shift_n_ts_target: Dict[int, int] = dict(
            zip(
                self.sampling_zooms_target.keys(),
                [int(v.get("shift_n_ts_target", 0)) for v in self.sampling_zooms_target.values()],
            )
        )

        self.p_dropout_all: float = p_dropout_all

        all_source_files: List[str] = []
        for data in self.data_dict['source'].values():
            if isinstance(data['files'], list) or isinstance(data['files'], ListConfig):
                all_source_files += data['files']
            else:
                all_source_files.append(data['files'])

        all_target_files: List[str] = []
        for data in self.data_dict['target'].values():
            if isinstance(data['files'], list) or isinstance(data['files'], ListConfig):
                all_target_files += data['files']
            else:
                all_target_files.append(data['files'])
        
        self.zoom_patch_sample: List[int] = [v['zoom_patch_sample'] for v in self.sampling_zooms.values()]
        self.zoom_time_steps_past: List[int] = [v['n_past_ts'] for v in self.sampling_zooms.values()]
        self.zoom_time_steps_future: List[int] = [v['n_future_ts'] for v in self.sampling_zooms.values()]
        
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
        source_ref = self.data_dict['source'].get(self.max_zoom_source) or self.data_dict['source'].get(
            str(self.max_zoom_source)
        )
        for k, file in enumerate(source_ref['files']):
            with xr.open_zarr(file, consolidated=False) as ds:
                self.time_steps_files.append(len(ds.time))

        # Build index map of (file, time window, region) per zoom.
        # Store index maps as compact numpy arrays to reduce Python object overhead.
        # Each row is: [file_idx, region_idx, t0, t1, ..., t(load_n_samples_time-1)]
        self.index_map: Dict[int, List[List[int]]] = dict(
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

                        row = [int(file_idx), int(region_idx_zoom)] + [int(t) for t in time_entry]
                        self.index_map[zoom].append(row)
                
        self.index_map = {
            z: np.asarray(idx_map, dtype=np.int32) for z, idx_map in self.index_map.items()
        }

        # Build variable groups from sampling configuration.
        fallback_variables = variables if variables is not None else self.data_dict.get("variables", {})
        self.groups_source_by_zoom: Dict[int, Dict[str, List[str]]] = self._resolve_groups_from_sampling(
            self.sampling_zooms_source,
            fallback_variables,
            side="source",
        )
        self.groups_target_by_zoom: Dict[int, Dict[str, List[str]]] = self._resolve_groups_from_sampling(
            self.sampling_zooms_target,
            fallback_variables,
            side="target",
        )
        self.variables_source_groups: Dict[str, List[str]] = self._validate_consistent_groups_across_zooms(
            self.groups_source_by_zoom,
            side="source",
        )
        self.variables_target_groups: Dict[str, List[str]] = self._validate_consistent_groups_across_zooms(
            self.groups_target_by_zoom,
            side="target",
        )

        if set(self.variables_source_groups.keys()) != set(self.variables_target_groups.keys()):
            raise ValueError(
                "Source and target groups must use the same group keys. "
                f"source={list(self.variables_source_groups.keys())}, "
                f"target={list(self.variables_target_groups.keys())}."
            )

        all_variables_source: List[str] = []
        all_variables_target: List[str] = []
        variable_ids: Dict[str, np.ndarray] = {}
        all_ids: List[int] = []
        self.group_ids: Dict[str, int] = {}
        offset = 0
        for group_id, group in enumerate(self.variables_source_groups.keys()):
            vars_source = list(self.variables_source_groups[group])
            vars_target = list(self.variables_target_groups[group])

            all_variables_source += vars_source
            all_variables_target += vars_target
            variable_ids[group] = np.arange(len(vars_source)) + offset
            all_ids = all_ids+list(variable_ids[group])
            offset = len(variable_ids[group])
            self.group_ids[group] = (group_id)

        self.all_variable_ids: Dict[str, int] = dict(zip(all_variables_source, all_ids))

        all_variables = list(dict.fromkeys(all_variables_source + all_variables_target))

        target_ref = self.data_dict['target'].get(self.max_zoom_target) or self.data_dict['target'].get(
            str(self.max_zoom_target)
        )
        if target_ref is None:
            target_ref = source_ref

        source_probe_file = source_ref['files'][0]
        target_probe_file = target_ref['files'][0]

        def _infer_grid_type(
            variable: str,
            primary_ds: xr.Dataset,
            primary_side: str,
            primary_file: str,
            fallback_ds: Optional[xr.Dataset] = None,
            fallback_side: Optional[str] = None,
            fallback_file: Optional[str] = None,
        ) -> Optional[str]:
            try:
                resolved_primary = self._get_resolved_variable_map(
                    ds=primary_ds,
                    variables=[variable],
                    side=primary_side,
                    file_path=primary_file,
                )[variable]
                return get_grid_type_from_var(primary_ds, resolved_primary)
            except (KeyError, ValueError):
                pass

            if fallback_ds is not None and fallback_side is not None and fallback_file is not None:
                resolved_fallback = self._get_resolved_variable_map(
                    ds=fallback_ds,
                    variables=[variable],
                    side=fallback_side,
                    file_path=fallback_file,
                )[variable]
                return get_grid_type_from_var(fallback_ds, resolved_fallback)

            available_primary = list(primary_ds.data_vars.keys())
            available_fallback = list(fallback_ds.data_vars.keys()) if fallback_ds is not None else []
            raise KeyError(
                f"Could not infer grid type for variable '{variable}'. "
                f"Tried source aliases {self._get_alias_candidates(variable, 'source')} and "
                f"target aliases {self._get_alias_candidates(variable, 'target')}. "
                f"Source probe vars: {available_primary[:10]}{'...' if len(available_primary) > 10 else ''}. "
                f"Target probe vars: {available_fallback[:10]}{'...' if len(available_fallback) > 10 else ''}."
            )

        with xr.open_zarr(source_probe_file, consolidated=False) as ds_source_probe:
            if target_probe_file == source_probe_file:
                ds_target_probe = ds_source_probe
                close_target_probe = False
            else:
                ds_target_probe = xr.open_zarr(target_probe_file, consolidated=False)
                close_target_probe = True

            try:
                self.vars_grid_types: Dict[str, Any] = {}
                for var in all_variables_source:
                    if var not in self.vars_grid_types:
                        self.vars_grid_types[var] = _infer_grid_type(
                            var,
                            ds_source_probe,
                            primary_side="source",
                            primary_file=source_probe_file,
                            fallback_ds=ds_target_probe,
                            fallback_side="target",
                            fallback_file=target_probe_file,
                        )

                for var in all_variables_target:
                    if var not in self.vars_grid_types:
                        self.vars_grid_types[var] = _infer_grid_type(
                            var,
                            ds_target_probe,
                            primary_side="target",
                            primary_file=target_probe_file,
                            fallback_ds=ds_source_probe,
                            fallback_side="source",
                            fallback_file=source_probe_file,
                        )
            finally:
                if close_target_probe:
                    ds_target_probe.close()

        self.grid_types: np.ndarray = np.unique(list(self.vars_grid_types.values()))
        self.grid_types_vars: Dict[Any, List[str]] = invert_dict(self.vars_grid_types)

        unique_source_files = np.unique(np.array(all_source_files))
        unique_target_files = np.unique(np.array(all_target_files))

        self.single_source: bool = len(unique_source_files) == 1
        self.single_target: bool = len(unique_target_files) == 1
        same_source_target_files = np.array_equal(unique_source_files, unique_target_files)
        self.mapping: Dict[int, Dict[Any, Any]] = {}
        if self.single_source and same_source_target_files:
            # Only reuse a single shared mapping when source and target reference
            # the same underlying file set.
            with xr.open_zarr(source_ref['files'][0], consolidated=False) as ds:
                mapping_hr = {}
                for grid_type in self.grid_types:
                    coords = get_coords_as_tensor(ds, grid_type=grid_type)
                    if coords is None:
                        continue
                    mapping_hr[grid_type] = mapping_fcn(coords, max(self.zooms))[max(self.zooms)]
            self.mapping[max(self.zooms)] = mapping_hr
        else:
            for zoom in self.zooms:
                # Build per-zoom mappings so source/target zooms can use different files/grids.
                mapping_grid_type = {}
                source_entry = (
                    self.data_dict['source'].get(zoom)
                    or self.data_dict['source'].get(str(zoom))
                    or self.data_dict['target'].get(zoom)
                    or self.data_dict['target'].get(str(zoom))
                )
                if source_entry is None:
                    continue
                with xr.open_zarr(source_entry['files'][0], consolidated=False) as ds:
                    for grid_type in self.grid_types:
                        coords = get_coords_as_tensor(ds, grid_type=grid_type)
                        if coords is None:
                            continue
                        mapping_grid_type[grid_type] = mapping_fcn(coords, zoom)[zoom]
                self.mapping[zoom] = mapping_grid_type

        self.load_once: bool = (
            unique_time_steps_past
            and unique_time_steps_future
            and unique_zoom_patch_sample
            and self.single_source
            and self.single_target
        )

        with open(norm_dict) as json_file:
            norm_dict = json.load(json_file)

        self.var_normalizers: Dict[int, Dict[str, Any]] = {}
        for zoom in self.zooms:
            self.var_normalizers[zoom] = {}
            for var in all_variables:
                norm_key = self._resolve_norm_key(var, norm_dict)
                if str(zoom) in norm_dict[norm_key].keys():
                    # Zoom-specific stats override global stats when available.
                    norm_class = norm_dict[norm_key][str(zoom)]['normalizer']['class']
                    assert norm_class in normalizers.__dict__.keys(), f'normalizer class {norm_class} not defined'
                    self.var_normalizers[zoom][var] = normalizers.__getattribute__(norm_class)(
                        norm_dict[norm_key][str(zoom)]['stats'],
                        norm_dict[norm_key][str(zoom)]['normalizer'])
                else:
                    norm_class = norm_dict[norm_key]['normalizer']['class']
                    assert norm_class in normalizers.__dict__.keys(), f'normalizer class {norm_class} not defined'
                    self.var_normalizers[zoom][var] = normalizers.__getattribute__(norm_class)(
                        norm_dict[norm_key]['stats'],
                        norm_dict[norm_key]['normalizer'])
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
            ds_source = xr.open_zarr(file_path_source, decode_times=False, consolidated=False)
        else:
            ds_source = xr.load_zarr(file_path_source, decode_times=False, consolidated=False)

        if file_path_target is None:
            ds_target = None

        elif file_path_target == file_path_source and not drop_source:
            ds_target = None
            
        else:
            if self.lazy_load:
                ds_target = xr.open_zarr(file_path_target, decode_times=False, consolidated=False)
            else:
                ds_target = xr.load_zarr(file_path_target, decode_times=False, consolidated=False)

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
        patch_indices = self.get_indices_from_patch_idx(zoom, patch_idx)

        for grid_type, variables_grid_type in self.grid_types_vars.items():
            mapping_grid = mapping.get(grid_type)
            if mapping_grid is None:
                # Current dataset/zoom may not define all global grid types.
                continue

            # Resolve indices either on the target grid (post-map) or the source grid (pre-map).
            post_map = mapping_zoom > zoom or (patch_dim is None and mapping_zoom >= zoom)
            if post_map:
                indices = mapping_grid['indices'][..., [0]].reshape(-1, 4 ** (mapping_zoom - zoom))
                if patch_dim:
                    isel_dict[patch_dim] = indices.view(-1)

            else:
                indices = mapping_grid['indices'][..., [0]]

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
        resolved_variable_names: Optional[Mapping[str, str]] = None,
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
        :param resolved_variable_names: Optional mapping from canonical variable names
            in ``variables_sample`` to actual dataset variable names in ``ds``.
        :return: Tuple ``(data_g, data_time, drop_mask)`` where ``data_g`` is a tensor of
            shape ``(v, t, n, d, f)`` (matching the ``(b, v, t, n, d, f)`` base shape with
            ``b`` handled by the caller), ``data_time`` is a tensor of shape ``(t,)``,
            and ``drop_mask`` is a tensor of shape ``(v, t, n, d, 1)``.
        """
        # Fetch raw patch indices
        drop_mask_ = drop_mask.clone() if drop_mask is not None else None
        if drop_mask_ is not None:
            # Accept (t, n), (v, t, n), or (1, v, t, n), then normalize to (v, t, n).
            if drop_mask_.ndim == 4:
                if drop_mask_.shape[0] != 1:
                    raise ValueError(
                        f"Expected drop_mask leading batch dim to be 1, got shape {tuple(drop_mask_.shape)}"
                    )
                drop_mask_ = drop_mask_.squeeze(0)
            if drop_mask_.ndim == 2:
                drop_mask_ = drop_mask_.unsqueeze(0)
            if drop_mask_.ndim != 3:
                raise ValueError(
                    f"Expected drop_mask to have 2, 3, or 4 dims, got shape {tuple(drop_mask_.shape)}"
                )
            drop_mask_ = drop_mask_.to(dtype=torch.bool)

        patch_dim_candidates = [d for d in ds.dims if "cell" in d or "ncells" in d]
        patch_dim = patch_dim_candidates[0] if patch_dim_candidates else None

        data_g = []
        mask = None
        for grid_type, variables_grid_type in self.grid_types_vars.items():
            variables = [str(var) for var in variables_sample if str(var) in variables_grid_type]
            if not variables:
                continue
            dataset_variables = [
                str(resolved_variable_names.get(var, var)) if resolved_variable_names is not None else var
                for var in variables
            ]

            mapping_grid = mapping.get(grid_type)
            if mapping_grid is None:
                raise KeyError(
                    f"Missing mapping for grid type '{grid_type}' at zoom {zoom} "
                    f"required by variables {variables}. Available mappings: {list(mapping.keys())}"
                )

            patch_indices = self.get_indices_from_patch_idx(zoom, patch_idx)

            mask = get_mapping_weights(mapping_grid)[..., 0].view(1, 1, -1, 1, 1)

            # Map indices differently depending on whether we are projecting from a higher zoom.
            post_map = mapping_zoom > zoom or (patch_dim is None and mapping_zoom >= zoom)
            if post_map:
                indices = mapping_grid['indices'][..., [0]].reshape(-1, 4 ** (mapping_zoom - zoom))

            else:
                indices = mapping_grid['indices'][..., [0]]
                mask = mask[:, :, patch_indices]

                if drop_mask_ is not None:
                    drop_mask_ = drop_mask_[..., patch_indices]

            ds_variables = ds[dataset_variables]
            arr = ds_variables.to_array().to_numpy()
            if arr.dtype == np.float64:
                arr = arr.astype(np.float32, copy=False)
            # Normalize in raw array layout first.
            data_g = torch.from_numpy(arr)
            if self.normalize_data:
                for k, variable in enumerate(variables):
                    data_g[k] = self.var_normalizers[zoom][variable].normalize(data_g[k])

            # Shape all inputs to the shared convention: (v, t, n, d, f).
            if patch_dim is None:
                # Regular lon/lat input: flatten spatial dimensions to n while keeping optional level as d.
                if 'level' in ds_variables.dims or 'lev' in ds_variables.dims or 'depth' in ds_variables.dims:
                    # Expected raw shape: (v, t, level, lat, lon)
                    data_g = data_g.permute(0, 1, 3, 4, 2).reshape(
                        data_g.shape[0], data_g.shape[1], -1, data_g.shape[2]
                    )
                    data_g = data_g.unsqueeze(dim=-1)  # (v, t, n, d=level, f=1)
                else:
                    # Expected raw shape: (v, t, lat, lon)
                    data_g = data_g.reshape(data_g.shape[0], data_g.shape[1], -1)
                    data_g = data_g.unsqueeze(dim=-1).unsqueeze(dim=-1)  # (v, t, n, d=1, f=1)
            else:
                # HealPix / ICON-like 1D cell input.
                data_g = data_g.unsqueeze(dim=-1)
                if 'level' not in ds_variables.dims and 'lev' not in ds_variables.dims and 'depth' not in ds_variables.dims:
                    data_g = data_g.unsqueeze(dim=2)
                data_g = data_g.transpose(2, 3)

            if not patch_dim and post_map:
                data_g = data_g[:, :, indices.view(-1), :, :]


        if drop_mask_ is not None and mask.dtype != torch.bool:
            drop_mask_expanded = drop_mask_.unsqueeze(dim=-1).unsqueeze(dim=-1)
            mask = (1 - drop_mask_expanded.to(mask.dtype)) * mask
            mask = mask.expand_as(data_g)

        elif drop_mask_ is not None:
            mapping_mask = mask.expand_as(data_g)
            drop_mask_expanded = drop_mask_.unsqueeze(dim=-1).unsqueeze(dim=-1).expand_as(mapping_mask)
            mask = torch.logical_and(mapping_mask, torch.logical_not(drop_mask_expanded))

        else:
            mask = None

        # Treat NaNs as masked values and zero them out before zoom transforms.
        nan_mask = torch.isnan(data_g)
        if nan_mask.any():
            data_g = data_g.clone()
            data_g[nan_mask] = 0

            if mask is None:
                mask = torch.logical_not(nan_mask)
            elif mask.dtype == torch.bool:
                mask = torch.logical_and(mask, torch.logical_not(nan_mask))
            else:
                mask = mask * torch.logical_not(nan_mask).to(mask.dtype)
        
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
        sample_configs = self.sampling_zooms.copy()
        if not data_source:
            for key, value in patch_index_zooms.items():
                if key in sample_configs:
                    sample_configs[key]['patch_index'] = value
            return {}, {}, sample_configs, {}
        if data_target is None:
            data_target = {}
        else:
            data_target = {
                int(zoom): value for zoom, value in data_target.items() if value is not None
            }

        data_source = encode_zooms(data_source, sample_configs, patch_index_zooms)
        data_target = encode_zooms(data_target, sample_configs, patch_index_zooms)

        if not hr_dopout and self.p_dropout_all > 0:
            drop = False
            for zoom in sorted(data_source.keys()):

                if self.p_dropout_all_zooms.get(zoom, 0) > 0 and not drop:
                    drop = torch.rand(1) < self.p_dropout_all_zooms.get(zoom, 0)

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

                if mask_mapping_zooms[zoom] is None:
                    mask_mapping_zooms[zoom] = torch.zeros_like(data_source[zoom],dtype=bool)
                else:
                    mask_mapping_zooms[zoom] = rearrange(mask_mapping_zooms[zoom], 'v (b t) n d f -> b 1 t n 1 (v d f)', b=self.load_n_samples_time)
            else:
                data_source[zoom] = rearrange(data_source[zoom], 'v (b t) n d f -> b v t n d f', b=self.load_n_samples_time)

                if mask_mapping_zooms[zoom] is None:
                    mask_mapping_zooms[zoom] = torch.zeros_like(data_source[zoom],dtype=bool)
                else:
                    mask_mapping_zooms[zoom] = rearrange(mask_mapping_zooms[zoom], 'v (b t) n d f -> b v t n d f', b=self.load_n_samples_time)

        for zoom in data_target.keys():
            if self.variables_as_features:
                data_target[zoom] = rearrange(
                    data_target[zoom], 'v (b t) n d f -> b 1 t n 1 (v d f)', b=self.load_n_samples_time
                )
            else:
                data_target[zoom] = rearrange(
                    data_target[zoom], 'v (b t) n d f -> b v t n d f', b=self.load_n_samples_time
                )

        for key, value in patch_index_zooms.items():
            if key in sample_configs:
                sample_configs[key]['patch_index'] = value
        
        
        # Optionally mask the last timesteps and repeat or zero them out.
        if any([self.mask_n_last_ts_zooms.get(z, 0) > 0 for z in data_source.keys()]):
            for zoom in data_source.keys():
                mask_n_last_ts = self.mask_n_last_ts_zooms.get(zoom, 0)
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

        if self.output_max_zoom_only:
            if data_source:
                max_zoom_source = max(data_source.keys())
                data_source = decode_zooms(data_source, sample_configs, max_zoom_source)
                mask_mapping_zooms = {max_zoom_source: mask_mapping_zooms[max_zoom_source]}
            else:
                mask_mapping_zooms = {}

            if data_target:
                max_zoom_target = max(data_target.keys())
                data_target = decode_zooms(data_target, sample_configs, max_zoom_target)

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
        selected_vars_source = {}
        selected_vars_target = {}
  
        var_indices_source = {}
        var_indices_target = {}
        group_keys = list(self.variables_source_groups.keys())
        # Sample variables per group to build a compact input for this item.
        for group in group_keys:
            variables_source = list(self.variables_source_groups[group])
            variables_target = list(self.variables_target_groups[group])

            indices_source = np.arange(len(variables_source))
            indices_target = np.arange(len(variables_target))

            if self.n_sample_variables != -1:
                if len(variables_source) == len(variables_target):
                    sample_size = min(self.n_sample_variables, len(variables_source))
                    if sample_size != len(variables_source):
                        indices_source = np.random.choice(indices_source, sample_size, replace=False)
                    indices_target = indices_source.copy()
                else:
                    sample_size_source = min(self.n_sample_variables, len(variables_source))
                    sample_size_target = min(self.n_sample_variables, len(variables_target))
                    if sample_size_source != len(variables_source):
                        indices_source = np.random.choice(indices_source, sample_size_source, replace=False)
                    if sample_size_target != len(variables_target):
                        indices_target = np.random.choice(indices_target, sample_size_target, replace=False)

            var_indices_source[group] = indices_source
            var_indices_target[group] = indices_target
            selected_vars_source[group] = np.array(variables_source)[indices_source]
            selected_vars_target[group] = np.array(variables_target)[indices_target]
            

        hr_dopout = self.p_dropout > 0 and torch.rand(1) > (self.p_dropout_all)

        # Only build a global dropout mask when a single source ensures shared indexing.
        if self.single_source and hr_dopout:
            nt = 1 + self.max_time_step_future + self.max_time_step_past
            total_vars = sum(len(selected_vars_source[group]) for group in selected_vars_source.keys())
            drop_mask_input = self.get_mask(
                total_vars,
                nt,
                self.indices[max(self.zooms_source)].size,
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
        ds_source = None
        ds_target = None
        source_file_loaded = None
        target_file_loaded = None
        for zoom in self.zooms:
            row = self.index_map[zoom][index]
            file_index = int(row[0])
            patch_index = int(row[1])
            time_indices = row[2:].tolist()

            zoom_in_source = zoom in self.sampling_zooms_source
            zoom_in_target = zoom in self.sampling_zooms_target

            source_entry = self.data_dict['source'].get(zoom) or self.data_dict['source'].get(str(zoom))
            target_entry = self.data_dict['target'].get(zoom) or self.data_dict['target'].get(str(zoom))

            if zoom_in_source and source_entry is not None:
                source_file = source_entry['files'][file_index]
            elif zoom_in_source and self.single_source:
                source_ref = self.data_dict['source'].get(self.max_zoom_source) or self.data_dict['source'].get(
                    str(self.max_zoom_source)
                )
                source_file = source_ref['files'][file_index]
            else:
                source_file = None

            if zoom_in_target and target_entry is not None:
                target_file = target_entry['files'][file_index]
            elif zoom_in_target and self.single_target:
                target_ref = self.data_dict['target'].get(self.max_zoom_target) or self.data_dict['target'].get(
                    str(self.max_zoom_target)
                )
                target_file = target_ref['files'][file_index]
            else:
                target_file = None

            if source_file is None and zoom_in_source:
                raise KeyError(
                    f"Missing source files for zoom {zoom}. "
                    "Define source data for that zoom or provide a single shared source file."
                )
            if target_file is None and zoom_in_target:
                raise KeyError(
                    f"Missing target files for zoom {zoom}. "
                    "Define target data for that zoom or provide a single shared target file."
                )

            mapping_zoom_source = None
            if source_file is not None:
                with xr.open_zarr(source_file, consolidated=False) as ds:
                    if "cell" in ds.sizes:
                        mapping_zoom_source = get_zoom_from_npix(ds.sizes["cell"])
                    elif "ncells" in ds.sizes:
                        mapping_zoom_source = get_zoom_from_npix(ds.sizes["ncells"])
                    else:
                        mapping_zoom_source = zoom if zoom in self.mapping else max(self.mapping.keys())

                    if mapping_zoom_source is None or mapping_zoom_source not in self.mapping:
                        mapping_zoom_source = zoom if zoom in self.mapping else max(self.mapping.keys())

            mapping_zoom_target = None
            if target_file is not None:
                with xr.open_zarr(target_file, consolidated=False) as ds:
                    if "cell" in ds.sizes:
                        mapping_zoom_target = get_zoom_from_npix(ds.sizes["cell"])
                    elif "ncells" in ds.sizes:
                        mapping_zoom_target = get_zoom_from_npix(ds.sizes["ncells"])
                    else:
                        mapping_zoom_target = zoom if zoom in self.mapping else max(self.mapping.keys())

                    if mapping_zoom_target is None or mapping_zoom_target not in self.mapping:
                        mapping_zoom_target = zoom if zoom in self.mapping else max(self.mapping.keys())

            if not loaded or source_file != source_file_loaded or target_file != target_file_loaded:
                if ds_source is not None:
                    ds_source.close()
                    ds_source = None
                if ds_target is not None:
                    ds_target.close()
                    ds_target = None

                if source_file is not None:
                    ds_source, ds_target = self.get_files(
                        source_file,
                        file_path_target=target_file,
                        drop_source=self.p_dropout > 0,
                    )
                elif target_file is not None:
                    if self.lazy_load:
                        ds_target = xr.open_zarr(target_file, decode_times=False, consolidated=False)
                    else:
                        ds_target = xr.load_zarr(target_file, decode_times=False, consolidated=False)
                source_file_loaded = source_file
                target_file_loaded = target_file
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
                for indices in var_indices_source.values():
                    drop_mask_zoom_groups.append(drop_mask_zoom[indices].unsqueeze(0))
    
            start_times = np.array(time_indices) - self.sampling_zooms[zoom]['n_past_ts'] 
            end_times = np.array(time_indices) + self.sampling_zooms[zoom]['n_future_ts']

            time_indices = np.stack(
                [np.arange(s, e + 1) for s, e in zip(start_times, end_times)],
                axis=0
            ).reshape(-1)

            if ds_source is not None and mapping_zoom_source is not None:
                ds_source_zoom = self.select_ranges(
                    ds_source,
                    time_indices,
                    patch_index,
                    self.mapping[mapping_zoom_source],
                    mapping_zoom_source,
                    zoom,
                )
                data_time_zooms[zoom] = (
                    torch.from_numpy(ds_source_zoom.time.values)
                    .view(self.load_n_samples_time, -1)
                    .to(torch.float32)
                )
            else:
                ds_source_zoom = None

            if zoom_in_target and (
                ds_target is not None or (ds_source is not None and self.shift_n_ts_target.get(zoom, 0) > 0)
            ):
                ds_target_base = ds_source if ds_target is None else ds_target
                mapping_zoom_target_ = mapping_zoom_source if mapping_zoom_target is None else mapping_zoom_target
                ds_target_zoom = self.select_ranges(
                    ds_target_base,
                    time_indices + self.shift_n_ts_target.get(zoom,0),
                    patch_index,
                    self.mapping[mapping_zoom_target_],
                    mapping_zoom_target_,
                    zoom,
                )
                if zoom not in data_time_zooms:
                    data_time_zooms[zoom] = (
                        torch.from_numpy(ds_target_zoom.time.values)
                        .view(self.load_n_samples_time, -1)
                        .to(torch.float32)
                    )
            else:
                ds_target_zoom = None

            source_var_name_map: Optional[Dict[str, str]] = None
            if zoom_in_source and ds_source is not None and source_file is not None:
                selected_source_vars_flat = [
                    str(var_name)
                    for group_vars in selected_vars_source.values()
                    for var_name in group_vars
                ]
                source_var_name_map = self._get_resolved_variable_map(
                    ds=ds_source,
                    variables=selected_source_vars_flat,
                    side="source",
                    file_path=source_file,
                )

            target_var_name_map: Optional[Dict[str, str]] = None
            if zoom_in_target and ds_target_zoom is not None:
                ds_target_base = ds_source if ds_target is None else ds_target
                if ds_target_base is None:
                    raise RuntimeError(
                        f"Internal error while resolving target aliases at zoom {zoom}: "
                        "target dataset is expected but missing."
                    )
                target_file_for_alias = target_file if target_file is not None else source_file
                if target_file_for_alias is None:
                    raise RuntimeError(
                        f"Internal error while resolving target aliases at zoom {zoom}: "
                        "no source or target file path available."
                    )
                selected_target_vars_flat = [
                    str(var_name)
                    for group_vars in selected_vars_target.values()
                    for var_name in group_vars
                ]
                target_var_name_map = self._get_resolved_variable_map(
                    ds=ds_target_base,
                    variables=selected_target_vars_flat,
                    side="target",
                    file_path=target_file_for_alias,
                )

            for group_idx, group in enumerate(group_keys):
                if zoom_in_source and ds_source_zoom is not None and mapping_zoom_source is not None:
                    data_source, drop_mask_zoom_group = self.get_data(
                        ds_source_zoom,
                        patch_index,
                        selected_vars_source[group],
                        self.mapping[mapping_zoom_source],
                        mapping_zoom_source,
                        zoom,
                        drop_mask=drop_mask_zoom_groups[group_idx],
                        resolved_variable_names=source_var_name_map,
                    )
                else:
                    data_source = None
                    drop_mask_zoom_group = None

                if zoom_in_target and ds_target_zoom is not None:
                    mapping_zoom_target_ = mapping_zoom_source if mapping_zoom_target is None else mapping_zoom_target
                    data_target,  _ = self.get_data(
                        ds_target_zoom,
                        patch_index,
                        selected_vars_target[group],
                        self.mapping[mapping_zoom_target_],
                        mapping_zoom_target_,
                        zoom,
                        resolved_variable_names=target_var_name_map,
                    )
                else:
                    data_target = None

                if data_source is not None:
                    source_zooms_groups[group_idx][zoom] = data_source
                    mask_mapping_zooms_groups[group_idx][zoom] = drop_mask_zoom_group
                if data_target is not None:
                    target_zooms_groups[group_idx][zoom] = data_target

            patch_index_zooms[zoom] = torch.tensor(patch_index)
            
        
        if ds_source is not None:
            ds_source.close()
        if ds_target is not None:
            ds_target.close()

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
                emb_group['VariableEmbedder'] = (
                    torch.tensor(list(var_indices_source[group]))
                    .view(1, -1)
                    .repeat_interleave(self.load_n_samples_time, dim=0)
                )
                emb_group['MGEmbedder'] = emb_group['VariableEmbedder']

                if StaticVariableEmbedder is not None:
                    emb_group['StaticVariableEmbedder'] = StaticVariableEmbedder

                    
                emb_group['TimeEmbedder'] = data_time_zooms
                emb_groups.append(emb_group)
        
            source_zooms_groups_out_ = {}
            target_zooms_groups_out_ = {}
            mask_zooms_groups_ = {}

        if self.variables_as_features:
            source_zoom_keys = list(source_zooms_groups_out[0].keys()) if source_zooms_groups_out else []
            target_zoom_keys = list(target_zooms_groups_out[0].keys()) if target_zooms_groups_out else []

            for zoom in source_zoom_keys:
                source_zooms_groups_out_[zoom] = torch.concat(
                    [group[zoom] for group in source_zooms_groups_out], dim=-1
                )
                mask_zooms_groups_[zoom] = torch.concat(
                    [group[zoom] for group in mask_zooms_groups], dim=-1
                )

            for zoom in target_zoom_keys:
                target_zooms_groups_out_[zoom] = torch.concat(
                    [group[zoom] for group in target_zooms_groups_out], dim=-1
                )

            feature_width = 0
            if source_zoom_keys:
                feature_width = int(source_zooms_groups_out_[source_zoom_keys[0]].shape[-1])

            emb = {#'StaticVariableEmbedder': None,#emb_groups[0]['StaticVariableEmbedder'],
                    'TimeEmbedder': emb_groups[0]['TimeEmbedder'],
                    'VarialeEmbedder': torch.zeros(feature_width, dtype=torch.long)}

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
