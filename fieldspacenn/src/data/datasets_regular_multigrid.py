"""Native Xarray loader for square regular multigrid data."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import xarray as xr
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.utils.data import Dataset

from . import normalizer as normalizers
from .datasets_base import _normalize_variables_config, _resolve_global_variable_ids
from ..modules.grids.grid_utils import encode_zooms
from ..modules.grids.regular_grid import (
    build_mean_pyramid,
    grid_to_morton,
    make_loss_region_masks,
    side_from_zoom,
    validate_regular_shape,
)


def _as_plain_mapping(value: Any) -> Any:
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _files(entry: Mapping[str, Any]) -> List[str]:
    value = entry.get("files")
    if isinstance(value, (str, Path)):
        return [str(value)]
    if isinstance(value, (list, tuple, ListConfig)):
        return [str(path) for path in value]
    raise ValueError(f"Expected a file or file list, got {value!r}.")


def _open_xarray(path: str, *, decode_times: bool) -> xr.Dataset:
    """Open either a NetCDF-like dataset or a Zarr store."""
    path_obj = Path(path)
    if path_obj.suffix.lower() == ".zarr" or (path_obj.is_dir() and (path_obj / ".zgroup").exists()):
        return xr.open_zarr(path, decode_times=decode_times)
    return xr.open_dataset(path, decode_times=decode_times)


def _normalize_zoom_sources(value: Mapping[Any, Any]) -> Dict[int, Dict[str, Any]]:
    value = _as_plain_mapping(value)
    if "files" in value:
        return {-1: dict(value)}
    return {int(zoom): dict(entry) for zoom, entry in value.items()}


class RegularMultigridDataset(Dataset):
    """Load native square regular grids and expose the multigrid batch contract.

    Model tensors use ``(sample_batch, variable, time, morton_cell, depth, feature)``.
    A configured image channel dimension maps to the final feature axis.
    """

    grid_type = "regular"
    root_cell_count = 4

    def __init__(
        self,
        data_dict: Mapping[str, Any],
        sampling_zooms: Mapping[int, Mapping[str, Any]],
        spatial_dims: Sequence[str],
        pyramid_mode: str = "derive",
        sample_dim: Optional[str] = None,
        time_dim: Optional[str] = None,
        channel_dim: Optional[str] = None,
        norm_dict: Optional[str] = None,
        normalize_data: bool = True,
        is_training: bool = True,
        train_loss_patch_zoom: int = -1,
        sampling_zooms_collate: Optional[Mapping[int, Mapping[str, Any]]] = None,
        sampling_zooms_target: Optional[Mapping[int, Mapping[str, Any]]] = None,
        load_n_samples_time: int = 1,
        target_time_shift: int = 0,
        deterministic: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if len(spatial_dims) != 2 or spatial_dims[0] == spatial_dims[1]:
            raise ValueError("spatial_dims must contain distinct [y, x] dimension names.")
        if pyramid_mode not in {"derive", "precomputed"}:
            raise ValueError("pyramid_mode must be either `derive` or `precomputed`.")
        if int(load_n_samples_time) != 1:
            raise ValueError(
                "RegularMultigridDataset currently requires load_n_samples_time=1; "
                "use DataLoader batching for independent images/sequences."
            )

        self.data_dict = copy.deepcopy(_as_plain_mapping(data_dict))
        self.sampling_zooms = {
            int(zoom): copy.deepcopy(dict(config))
            for zoom, config in _as_plain_mapping(sampling_zooms).items()
        }
        self.sampling_zooms_target = {
            int(zoom): copy.deepcopy(dict(config))
            for zoom, config in _as_plain_mapping(
                sampling_zooms if sampling_zooms_target is None else sampling_zooms_target
            ).items()
        }
        self.sampling_zooms_collate = (
            None
            if sampling_zooms_collate is None
            else {
                int(zoom): copy.deepcopy(dict(config))
                for zoom, config in _as_plain_mapping(sampling_zooms_collate).items()
            }
        )
        self.zooms = sorted(self.sampling_zooms)
        if not self.zooms:
            raise ValueError("sampling_zooms must not be empty.")
        for zoom, config in self.sampling_zooms.items():
            if int(config.get("zoom_patch_sample", -1)) != -1:
                raise ValueError(
                    "Regular model inputs must remain full-grid: set every "
                    "sampling_zooms.*.zoom_patch_sample to -1 and use "
                    "train_loss_patch_zoom for supervision patches."
                )
        if train_loss_patch_zoom >= 0 and train_loss_patch_zoom > min(self.sampling_zooms_target):
            raise ValueError(
                "train_loss_patch_zoom must not exceed the coarsest target zoom."
            )

        self.spatial_dims = (str(spatial_dims[0]), str(spatial_dims[1]))
        self.sample_dim = None if sample_dim is None else str(sample_dim)
        self.time_dim = None if time_dim is None else str(time_dim)
        self.channel_dim = None if channel_dim is None else str(channel_dim)
        named_dims = [dim for dim in (*self.spatial_dims, self.sample_dim, self.time_dim, self.channel_dim) if dim]
        if len(named_dims) != len(set(named_dims)):
            raise ValueError("spatial, sample, time, and channel dimensions must be distinct.")
        self.pyramid_mode = pyramid_mode
        self.normalize_data = bool(normalize_data)
        self.is_training = bool(is_training)
        self.train_loss_patch_zoom = int(train_loss_patch_zoom)
        self.load_n_samples_time = 1
        self.target_time_shift = int(target_time_shift)
        self.deterministic = bool(deterministic)
        self.apply_diff = True
        self.max_zoom = max(self.zooms)

        variables_by_group, explicit_ids, _ = _normalize_variables_config(self.data_dict["variables"])
        self.variables_by_group = {
            name: variables for name, variables in variables_by_group.items()
            if name not in {"embedding", "embedding_1D"}
        }
        if not self.variables_by_group:
            raise ValueError("At least one spatial variable group is required.")
        self.all_variable_ids = _resolve_global_variable_ids(self.variables_by_group, explicit_ids)
        self.variables_target_groups = copy.deepcopy(self.variables_by_group)

        self.source_entries = _normalize_zoom_sources(self.data_dict["source"])
        self.target_entries = _normalize_zoom_sources(self.data_dict.get("target", self.data_dict["source"]))
        self._validate_source_configuration(self.source_entries, "source")
        self._validate_source_configuration(self.target_entries, "target")

        anchor_source = self._entry_for_zoom(self.source_entries, self.max_zoom)
        anchor_target = self._entry_for_zoom(self.target_entries, self.max_zoom)
        self.source_files = _files(anchor_source)
        self.target_files = _files(anchor_target)
        if len(self.source_files) != len(self.target_files):
            raise ValueError("Source and target file lists must have identical lengths.")
        self._validate_file_alignment()

        self.var_normalizers = self._build_normalizers(norm_dict)
        self.index_map: List[Tuple[int, int, int]] = []
        self._build_index_map()
        self.len_dataset = len(self.index_map)
        self.sample_time_length_signatures = [
            tuple(
                value
                for zoom in self.zooms
                for value in (
                    self._window_length(self.sampling_zooms[zoom]),
                    self._window_length(self.sampling_zooms_target[zoom]),
                )
            )
            for _ in self.index_map
        ]

    @staticmethod
    def _window_length(config: Mapping[str, Any]) -> int:
        return int(config.get("n_past_ts", 0)) + int(config.get("n_future_ts", 0)) + 1

    def _validate_source_configuration(self, entries: Mapping[int, Any], name: str) -> None:
        if self.pyramid_mode == "derive":
            if self.max_zoom not in entries and -1 not in entries:
                raise ValueError(f"Derived {name} data requires a maximum-zoom file entry.")
        else:
            missing = [zoom for zoom in self.zooms if zoom not in entries]
            if missing:
                raise ValueError(f"Precomputed {name} data is missing zooms {missing}.")
            counts = {len(_files(entries[zoom])) for zoom in self.zooms}
            if len(counts) != 1:
                raise ValueError(f"Precomputed {name} zooms must have equal file counts.")

    @staticmethod
    def _entry_for_zoom(entries: Mapping[int, Dict[str, Any]], zoom: int) -> Dict[str, Any]:
        if zoom in entries:
            return entries[zoom]
        if -1 in entries:
            return entries[-1]
        raise KeyError(f"No data entry for zoom {zoom}.")

    def _build_normalizers(self, norm_dict_path: Optional[str]) -> Dict[int, Dict[str, Any]]:
        result = {zoom: {} for zoom in self.zooms}
        if not self.normalize_data or norm_dict_path is None:
            return result
        with open(norm_dict_path, encoding="utf-8") as stream:
            definitions = json.load(stream)
        for zoom in self.zooms:
            for variables in self.variables_by_group.values():
                for variable in variables:
                    if variable not in definitions:
                        raise KeyError(f"No normalization definition for `{variable}`.")
                    definition = definitions[variable]
                    if str(zoom) in definition:
                        definition = definition[str(zoom)]
                    normalizer_config = definition["normalizer"]
                    class_name = normalizer_config["class"]
                    if not hasattr(normalizers, class_name):
                        raise ValueError(f"Unknown normalizer class `{class_name}`.")
                    result[zoom][variable] = getattr(normalizers, class_name)(
                        definition["stats"], normalizer_config
                    )
        return result

    def _validate_file_alignment(self) -> None:
        """Validate logical sample/time/channel axes across source, target, and zooms."""
        entry_sets = (("source", self.source_entries), ("target", self.target_entries))
        reference_signatures: Dict[int, Tuple[int, int, int]] = {}
        for label, entries in entry_sets:
            zooms = [self.max_zoom] if self.pyramid_mode == "derive" else self.zooms
            for zoom in zooms:
                paths = _files(self._entry_for_zoom(entries, zoom))
                for file_index, path in enumerate(paths):
                    with _open_xarray(path, decode_times=False) as dataset:
                        self._validate_dataset(dataset, zoom)
                        signature = (
                            int(dataset.sizes[self.sample_dim]) if self.sample_dim else 1,
                            int(dataset.sizes[self.time_dim]) if self.time_dim else 1,
                            int(dataset.sizes[self.channel_dim]) if self.channel_dim else 1,
                        )
                    if file_index not in reference_signatures:
                        reference_signatures[file_index] = signature
                    elif signature != reference_signatures[file_index]:
                        raise ValueError(
                            f"Logical file {file_index} has inconsistent sample/time/channel "
                            f"sizes at {label} zoom {zoom}: {signature} != "
                            f"{reference_signatures[file_index]}."
                        )

    def _build_index_map(self) -> None:
        requested_timesteps: Optional[set[int]] = None
        if "timesteps" in self.data_dict:
            requested_timesteps = set()
            for entry in self.data_dict["timesteps"]:
                if isinstance(entry, int) or "-" not in str(entry):
                    requested_timesteps.add(int(entry))
                else:
                    start, end = map(int, str(entry).split("-"))
                    requested_timesteps.update(range(start, end))

        for file_index, path in enumerate(self.source_files):
            with _open_xarray(path, decode_times=False) as dataset:
                self._validate_dataset(dataset, self.max_zoom)
                n_samples = int(dataset.sizes[self.sample_dim]) if self.sample_dim else 1
                n_times = int(dataset.sizes[self.time_dim]) if self.time_dim else 1
            max_past = max(
                max(int(config.get("n_past_ts", 0)) for config in configs.values())
                for configs in (self.sampling_zooms, self.sampling_zooms_target)
            )
            max_future = max(
                max(int(config.get("n_future_ts", 0)) for config in configs.values())
                for configs in (self.sampling_zooms, self.sampling_zooms_target)
            )
            first_time = max(max_past, max_past - self.target_time_shift)
            last_time = min(
                n_times - max_future,
                n_times - max_future - self.target_time_shift,
            )
            for sample_index in range(n_samples):
                for time_index in range(first_time, last_time):
                    if requested_timesteps is None or time_index in requested_timesteps:
                        self.index_map.append((file_index, sample_index, time_index))
        if not self.index_map:
            raise ValueError("No samples remain after applying dimension and time-window constraints.")

    def _validate_dataset(self, dataset: xr.Dataset, zoom: int) -> None:
        y_dim, x_dim = self.spatial_dims
        if y_dim not in dataset.sizes or x_dim not in dataset.sizes:
            raise ValueError(f"Dataset must contain spatial dimensions {self.spatial_dims}.")
        actual_zoom = validate_regular_shape(dataset.sizes[y_dim], dataset.sizes[x_dim])
        if actual_zoom != int(zoom):
            raise ValueError(
                f"Configured zoom {zoom} requires {side_from_zoom(zoom)}x{side_from_zoom(zoom)}, "
                f"but the dataset is zoom {actual_zoom}."
            )
        for optional_dim in (self.sample_dim, self.time_dim, self.channel_dim):
            if optional_dim is not None and optional_dim not in dataset.sizes:
                raise ValueError(f"Configured dimension `{optional_dim}` is missing.")

    def _path_for(self, entries: Mapping[int, Dict[str, Any]], zoom: int, file_index: int) -> str:
        paths = _files(self._entry_for_zoom(entries, zoom))
        return paths[file_index]

    def _read_variable(
        self,
        path: str,
        variable: str,
        zoom: int,
        sample_index: int,
        time_index: int,
        window_config: Mapping[str, Any],
        apply_normalization: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        with _open_xarray(path, decode_times=True) as dataset:
            self._validate_dataset(dataset, zoom)
            if variable not in dataset:
                raise KeyError(f"Variable `{variable}` is missing from {path}.")
            array = dataset[variable]
            selectors: Dict[str, Any] = {}
            if self.sample_dim:
                selectors[self.sample_dim] = sample_index
            if self.time_dim:
                past = int(window_config.get("n_past_ts", 0))
                future = int(window_config.get("n_future_ts", 0))
                selectors[self.time_dim] = slice(time_index - past, time_index + future + 1)
            array = array.isel(selectors)

            allowed = set(self.spatial_dims)
            if self.time_dim:
                allowed.add(self.time_dim)
            if self.channel_dim:
                allowed.add(self.channel_dim)
            unexpected = [dimension for dimension in array.dims if dimension not in allowed]
            if unexpected:
                raise ValueError(
                    f"Variable `{variable}` has unsupported dimensions {unexpected}; "
                    "only configured sample/time/spatial/channel dimensions are allowed."
                )
            order = []
            if self.time_dim:
                order.append(self.time_dim)
            order.extend(self.spatial_dims)
            if self.channel_dim:
                order.append(self.channel_dim)
            values = np.asarray(array.transpose(*order).values)
            if not self.time_dim:
                values = values[None, ...]
            if not self.channel_dim:
                values = values[..., None]
            if not np.isfinite(values).all():
                raise ValueError(f"Variable `{variable}` in {path} contains NaN or infinity.")
            # Copy out of xarray's potentially read-only backing store because
            # residual zoom encoding updates its working tensors in place.
            tensor = torch.tensor(np.array(values, copy=True), dtype=torch.float32)

            time_values = None
            if self.time_dim:
                coordinate = array.coords.get(self.time_dim)
                if coordinate is None:
                    past = int(window_config.get("n_past_ts", 0))
                    future = int(window_config.get("n_future_ts", 0))
                    time_values = torch.arange(
                        time_index - past,
                        time_index + future + 1,
                        dtype=torch.float32,
                    )
                else:
                    raw_time = np.asarray(coordinate.values)
                    if np.issubdtype(raw_time.dtype, np.datetime64):
                        raw_time = raw_time.astype("datetime64[s]").astype(np.int64)
                    time_values = torch.tensor(np.array(raw_time, copy=True), dtype=torch.float32)

        if apply_normalization and variable in self.var_normalizers[zoom]:
            # Put image channels on the normalizer's conventional axis 1.
            channel_first = tensor.permute(0, 3, 1, 2)
            channel_first = self.var_normalizers[zoom][variable].normalize(channel_first)
            tensor = channel_first.permute(0, 2, 3, 1)
        return tensor, time_values

    def _load_zoom_variables(
        self,
        entries: Mapping[int, Dict[str, Any]],
        variables: Sequence[str],
        sample_index: int,
        time_index: int,
        file_index: int,
        configs: Mapping[int, Mapping[str, Any]],
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, Optional[torch.Tensor]]]:
        values: Dict[int, List[torch.Tensor]] = {zoom: [] for zoom in self.zooms}
        times: Dict[int, Optional[torch.Tensor]] = {zoom: None for zoom in self.zooms}
        if self.pyramid_mode == "derive":
            path = self._path_for(entries, self.max_zoom, file_index)
            for variable in variables:
                for zoom in self.zooms:
                    # Temporal windows may differ by zoom. Load the requested
                    # window at native resolution and pool only its spatial axes.
                    highest, time_values = self._read_variable(
                        path,
                        variable,
                        self.max_zoom,
                        sample_index,
                        time_index,
                        configs[zoom],
                        apply_normalization=False,
                    )
                    if zoom == self.max_zoom:
                        tensor = highest
                    else:
                        tensor = build_mean_pyramid(
                            highest,
                            list(range(zoom, self.max_zoom + 1)),
                            spatial_dims=(1, 2),
                        )[zoom]
                    if variable in self.var_normalizers[zoom]:
                        # Every derived level uses its own configured zoom statistics.
                        channel_first = tensor.permute(0, 3, 1, 2)
                        channel_first = self.var_normalizers[zoom][variable].normalize(channel_first)
                        tensor = channel_first.permute(0, 2, 3, 1)
                    values[zoom].append(tensor)
                    times[zoom] = time_values
        else:
            for zoom in self.zooms:
                path = self._path_for(entries, zoom, file_index)
                for variable in variables:
                    tensor, time_values = self._read_variable(
                        path, variable, zoom, sample_index, time_index, configs[zoom]
                    )
                    values[zoom].append(tensor)
                    times[zoom] = time_values

        output: Dict[int, torch.Tensor] = {}
        for zoom, variable_values in values.items():
            channel_counts = {value.shape[-1] for value in variable_values}
            if len(channel_counts) != 1:
                raise ValueError(
                    f"Variables in one group must have the same channel count at zoom {zoom}."
                )
            # T,H,W,C -> V,T,N,D=1,F=C, with a leading sample-batch axis.
            morton = [grid_to_morton(value, spatial_dims=(1, 2)) for value in variable_values]
            output[zoom] = torch.stack(morton, dim=0).unsqueeze(-2).unsqueeze(0)
        return output, times

    @staticmethod
    def _encode(
        zooms: Dict[int, torch.Tensor],
        configs: Mapping[int, Mapping[str, Any]],
    ) -> Dict[int, torch.Tensor]:
        configs = copy.deepcopy(configs)
        patch_indices = {zoom: torch.zeros(1, dtype=torch.long) for zoom in zooms}
        return encode_zooms(
            zooms,
            configs,
            patch_indices,
            root_cell_count=RegularMultigridDataset.root_cell_count,
        )

    def __getitem__(self, index: int):
        file_index, sample_index, time_index = self.index_map[index]
        source_groups = []
        target_groups = []
        attention_mask_groups = []
        embedding_groups = []
        loss_region_mask_groups = []

        random_patch_index = None
        if self.is_training and self.train_loss_patch_zoom >= 0:
            n_regions = self.root_cell_count * 4**self.train_loss_patch_zoom
            if self.deterministic:
                random_patch_index = int(index % n_regions)
            else:
                random_patch_index = int(torch.randint(n_regions, (1,)).item())

        for variables in self.variables_by_group.values():
            source, source_times = self._load_zoom_variables(
                self.source_entries,
                variables,
                sample_index,
                time_index,
                file_index,
                self.sampling_zooms,
            )
            target, _ = self._load_zoom_variables(
                self.target_entries,
                variables,
                sample_index,
                time_index + self.target_time_shift,
                file_index,
                self.sampling_zooms_target,
            )
            source = self._encode(source, self.sampling_zooms)
            target = self._encode(target, self.sampling_zooms_target)
            attention_masks = {zoom: torch.zeros_like(value, dtype=torch.bool) for zoom, value in source.items()}

            variable_ids = torch.tensor(
                [self.all_variable_ids[variable] for variable in variables], dtype=torch.long
            ).view(1, -1)
            embeddings: Dict[str, Any] = {
                "VariableEmbedder": variable_ids,
                "MGEmbedder": variable_ids,
                "PressureLevelEmbedder": torch.empty(0),
            }
            if self.time_dim:
                embeddings["TimeEmbedder"] = {
                    zoom: source_times[zoom].view(1, -1) for zoom in self.zooms
                }

            if random_patch_index is None:
                loss_masks: Dict[int, torch.Tensor] = {}
            else:
                loss_masks = make_loss_region_masks(
                    target,
                    self.train_loss_patch_zoom,
                    random_patch_index,
                    spatial_dim=3,
                )
            source_groups.append(source)
            target_groups.append(target)
            attention_mask_groups.append(attention_masks)
            embedding_groups.append(embeddings)
            loss_region_mask_groups.append(loss_masks)

        patch_indices = {zoom: torch.zeros(1, dtype=torch.long) for zoom in self.zooms}
        return (
            source_groups,
            target_groups,
            attention_mask_groups,
            embedding_groups,
            patch_indices,
            loss_region_mask_groups,
        )

    def __len__(self) -> int:
        return self.len_dataset
