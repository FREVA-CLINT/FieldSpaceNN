import os
import shutil
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional, Tuple

import healpy as hp
import hydra
import numpy as np
import torch
import xarray as xr
import zarr
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
from omegaconf import DictConfig, ListConfig, OmegaConf

from .data.datasets_base import BaseDataset
from .data.pl_data_module import DataModule
from .modules.grids.grid_utils import decode_zooms
from .utils.helpers import load_from_state_dict, merge_sampling_dicts


def _get_zoom_tensor(zoom_dict: Mapping[Any, torch.Tensor], zoom: int) -> torch.Tensor:
    if zoom in zoom_dict:
        return zoom_dict[zoom]
    zoom_str = str(zoom)
    if zoom_str in zoom_dict:
        return zoom_dict[zoom_str]
    raise KeyError(f"Zoom {zoom} not found. Available keys: {list(zoom_dict.keys())}")


def _grouped_variables(data_dict: Mapping[str, Any]) -> List[List[str]]:
    return [
        list(variables)
        for group_name, variables in data_dict["variables"].items()
        if group_name != "embedding"
    ]


def _flatten_batch_indices(batch_indices: Any) -> List[int]:
    if batch_indices is None:
        return []
    if torch.is_tensor(batch_indices):
        return [int(idx) for idx in batch_indices.detach().cpu().view(-1).tolist()]
    if isinstance(batch_indices, np.ndarray):
        return [int(idx) for idx in batch_indices.reshape(-1).tolist()]
    if isinstance(batch_indices, (list, tuple)):
        out: List[int] = []
        for item in batch_indices:
            out.extend(_flatten_batch_indices(item))
        return out
    return [int(batch_indices)]


def _as_numpy_1d(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().view(-1).numpy()
    return np.asarray(value).reshape(-1)


def _as_file_list(files: Any) -> List[str]:
    if isinstance(files, ListConfig):
        return [str(file) for file in OmegaConf.to_container(files, resolve=True)]
    if isinstance(files, (list, tuple)):
        return [str(file) for file in files]
    return [str(files)]


def _create_array(
    group: Any,
    name: str,
    shape: Tuple[int, ...],
    chunks: Tuple[int, ...],
    dtype: Any,
    fill_value: Any,
    dimension_names: Sequence[str],
) -> Any:
    kwargs = {
        "shape": shape,
        "chunks": chunks,
        "dtype": dtype,
        "fill_value": fill_value,
    }
    try:
        array = group.create_array(name, dimension_names=tuple(dimension_names), **kwargs)
    except TypeError:
        try:
            array = group.create_array(name, **kwargs)
        except AttributeError:
            array = group.create_dataset(name, **kwargs)
    except AttributeError:
        array = group.create_dataset(name, **kwargs)

    array.attrs["_ARRAY_DIMENSIONS"] = list(dimension_names)
    return array


class HealPixZarrPredictionWriter(BasePredictionWriter):
    def __init__(
        self,
        output_path: str,
        dataset: BaseDataset,
        zoom: int,
        n_autoregressive_steps: int,
        ckpt_path: Optional[str] = None,
        overwrite: bool = False,
        time_chunk: int = 1,
        prediction_timedelta_chunk: Optional[int] = None,
        convention: str = "weatherbench2",
    ) -> None:
        super().__init__(write_interval="batch")
        self.output_path = output_path
        self.dataset = dataset
        self.zoom = int(zoom)
        self.n_autoregressive_steps = int(n_autoregressive_steps)
        self.ckpt_path = ckpt_path
        self.overwrite = overwrite
        self.time_chunk = int(time_chunk)
        self.prediction_timedelta_chunk = int(prediction_timedelta_chunk or n_autoregressive_steps)
        self.convention = str(convention)

        self.grouped_variables = _grouped_variables(dataset.data_dict)
        self.variables = [var for group in self.grouped_variables for var in group]
        self.npix = int(hp.nside2npix(2 ** self.zoom))
        self.normalizer_zoom = max(dataset.var_normalizers.keys())
        self.combined_zoom = max(int(zoom) for zoom in dataset.sampling_zooms_target.keys())

        self._root = None
        self._arrays: Dict[str, Any] = {}
        self._combined_root = None
        self._combined_arrays: Dict[str, Any] = {}
        self._time_lookup, self._time_coord, self._time_file = self._build_time_index()
        self._source_attrs, self._level_values = self._read_source_metadata()
        self._time_step = self._infer_time_step()

    def _build_time_index(self) -> Tuple[Dict[Tuple[int, int], int], np.ndarray, np.ndarray]:
        time_lookup: "OrderedDict[Tuple[int, int], int]" = OrderedDict()
        rows = self.dataset.index_map[self.zoom]
        for row in rows:
            file_idx = int(row[0])
            for center_time_idx in row[2:]:
                key = (file_idx, int(center_time_idx))
                if key not in time_lookup:
                    time_lookup[key] = len(time_lookup)

        time_values_by_file = self._read_time_values()
        time_coord = np.zeros(len(time_lookup), dtype=np.int64)
        time_file = np.zeros(len(time_lookup), dtype=np.int32)
        for key, time_array_idx in time_lookup.items():
            file_idx, source_time_idx = key
            time_file[time_array_idx] = file_idx
            time_coord[time_array_idx] = time_values_by_file[file_idx][source_time_idx]

        return dict(time_lookup), time_coord, time_file

    def _source_files_for_zoom(self) -> Sequence[str]:
        source_dict = self.dataset.data_dict["source"]
        if self.zoom in source_dict:
            return _as_file_list(source_dict[self.zoom]["files"])
        zoom_str = str(self.zoom)
        if zoom_str in source_dict:
            return _as_file_list(source_dict[zoom_str]["files"])
        source_zoom = max(int(zoom) for zoom in source_dict.keys())
        source_key = source_zoom if source_zoom in source_dict else str(source_zoom)
        return _as_file_list(source_dict[source_key]["files"])

    def _read_time_values(self) -> Dict[int, np.ndarray]:
        time_values_by_file = {}
        for file_idx, source_file in enumerate(self._source_files_for_zoom()):
            with xr.open_dataset(source_file, decode_times=False) as ds:
                time_values_by_file[file_idx] = np.asarray(ds.time.values)
        return time_values_by_file

    def _read_source_metadata(self) -> Tuple[Dict[str, Dict[str, Any]], Dict[int, np.ndarray]]:
        source_attrs: Dict[str, Dict[str, Any]] = {}
        level_values: Dict[int, np.ndarray] = {}
        candidate_zooms = sorted({self.zoom, self.combined_zoom})
        for zoom in candidate_zooms:
            source_files = self._source_files_for_zoom_for_key(zoom)
            if not source_files:
                continue
            with xr.open_dataset(source_files[0], decode_times=False) as ds:
                for variable in self.variables:
                    if variable in ds and variable not in source_attrs:
                        source_attrs[variable] = dict(ds[variable].attrs)
                if "level" in ds.coords:
                    level_values[zoom] = np.asarray(ds["level"].values)

        return source_attrs, level_values

    def _infer_time_step(self) -> int:
        source_files = self._source_files_for_zoom()
        if not source_files:
            return 1
        with xr.open_dataset(source_files[0], decode_times=False) as ds:
            time_values = np.asarray(ds.time.values)
        if time_values.size < 2:
            return 1
        diffs = np.diff(time_values.astype(np.int64))
        if diffs.size == 0:
            return 1
        return int(np.median(diffs))

    def _source_files_for_zoom_for_key(self, zoom: int) -> Sequence[str]:
        source_dict = self.dataset.data_dict["source"]
        if zoom in source_dict:
            return _as_file_list(source_dict[zoom]["files"])
        zoom_str = str(zoom)
        if zoom_str in source_dict:
            return _as_file_list(source_dict[zoom_str]["files"])
        return self._source_files_for_zoom()

    def _init_group_store(
        self,
        group_root: Any,
        arrays: Dict[str, Any],
        prediction_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        zoom: int,
    ) -> None:
        npix = int(hp.nside2npix(2 ** zoom))

        prediction_timedelta = (
            np.arange(1, self.n_autoregressive_steps + 1, dtype=np.int64) * self._time_step
        )
        valid_time = self._time_coord[:, None] + prediction_timedelta[None, :]
        self._write_coordinate_to_group(group_root, "time", self._time_coord, ("time",))
        self._write_coordinate_to_group(group_root, "time_file", self._time_file, ("time",))
        self._write_coordinate_to_group(
            group_root,
            "prediction_timedelta",
            prediction_timedelta,
            ("prediction_timedelta",),
        )
        self._write_coordinate_to_group(
            group_root,
            "valid_time",
            valid_time,
            ("time", "prediction_timedelta"),
        )
        self._write_coordinate_to_group(group_root, "cell", np.arange(npix, dtype=np.int64), ("cell",))
        if zoom in self._level_values:
            self._write_coordinate_to_group(group_root, "level", self._level_values[zoom], ("level",))

        for group_pred, variables in zip(prediction_groups, self.grouped_variables):
            if not group_pred:
                continue
            group_tensor = _get_zoom_tensor(group_pred, zoom)
            if int(group_tensor.shape[1]) != len(variables):
                raise ValueError(
                    "Prediction variable dimension does not match configured variable groups. "
                    f"shape={tuple(group_tensor.shape)}, variables={variables}"
                )
            for local_var_idx, variable in enumerate(variables):
                arrays[variable] = self._create_variable_array_for_group(
                    group_root=group_root,
                    variable=variable,
                    sample_tensor=group_tensor[:, local_var_idx],
                    npix=npix,
                )

    def _init_store(self, prediction: Mapping[str, Any]) -> None:
        if self._root is not None:
            return
        if os.path.exists(self.output_path):
            if not self.overwrite:
                raise FileExistsError(
                    f"Rollout output path already exists: {self.output_path}. "
                    "Set rollout.overwrite=true to replace it."
                )
            shutil.rmtree(self.output_path)
        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        self._root = zarr.open_group(self.output_path, mode="w")
        self._root.attrs.update(
            {
                "healpix_nested": True,
                "healpix_zoom": self.zoom,
                "n_autoregressive_steps": self.n_autoregressive_steps,
                "ckpt_path": self.ckpt_path,
                "source": "fieldspacenn autoregressive rollout",
                "combined_group": "combined",
                "combined_zoom": self.combined_zoom,
                "convention": self.convention,
            }
        )
        self._init_group_store(
            group_root=self._root,
            arrays=self._arrays,
            prediction_groups=prediction["output"],
            zoom=self.zoom,
        )

        self._combined_root = self._root.create_group("combined")
        self._combined_root.attrs.update(
            {
                "healpix_nested": True,
                "healpix_zoom": self.combined_zoom,
                "n_autoregressive_steps": self.n_autoregressive_steps,
                "ckpt_path": self.ckpt_path,
                "source": "fieldspacenn autoregressive rollout combined",
                "convention": self.convention,
            }
        )
        self._init_group_store(
            group_root=self._combined_root,
            arrays=self._combined_arrays,
            prediction_groups=prediction["output_combined"],
            zoom=self.combined_zoom,
        )

    def _write_coordinate_to_group(self, group_root: Any, name: str, values: np.ndarray, dims: Sequence[str]) -> None:
        array = _create_array(
            group_root,
            name,
            shape=tuple(values.shape),
            chunks=tuple(values.shape) if values.shape else (),
            dtype=values.dtype,
            fill_value=0,
            dimension_names=dims,
        )
        array[...] = values

    def _create_variable_array_for_group(
        self,
        group_root: Any,
        variable: str,
        sample_tensor: torch.Tensor,
        npix: int,
    ) -> Any:
        _, n_lead, _, n_level, n_feature = sample_tensor.shape
        time_chunk = min(self.time_chunk, max(1, len(self._time_coord)))
        prediction_timedelta_chunk = min(self.prediction_timedelta_chunk, max(1, int(n_lead)))

        if int(n_feature) != 1:
            shape = (len(self._time_coord), int(n_lead), int(n_level), npix, int(n_feature))
            chunks = (time_chunk, prediction_timedelta_chunk, int(n_level), npix, int(n_feature))
            dims = ("time", "prediction_timedelta", "level", "cell", "feature")
        elif int(n_level) == 1:
            shape = (len(self._time_coord), int(n_lead), npix)
            chunks = (time_chunk, prediction_timedelta_chunk, npix)
            dims = ("time", "prediction_timedelta", "cell")
        else:
            shape = (len(self._time_coord), int(n_lead), int(n_level), npix)
            chunks = (time_chunk, prediction_timedelta_chunk, int(n_level), npix)
            dims = ("time", "prediction_timedelta", "level", "cell")

        array = _create_array(
            group_root,
            variable,
            shape=shape,
            chunks=chunks,
            dtype=np.float32,
            fill_value=np.nan,
            dimension_names=dims,
        )
        array.attrs.update(self._source_attrs.get(variable, {}))
        array.attrs["coordinates"] = "time prediction_timedelta valid_time cell"
        return array

    def _batch_time_indices(self, batch_indices: Any) -> np.ndarray:
        dataset_indices = _flatten_batch_indices(batch_indices)
        if not dataset_indices:
            raise ValueError("Lightning did not provide batch_indices; cannot place rollout output.")

        time_indices: List[int] = []
        rows = self.dataset.index_map[self.zoom]
        for dataset_idx in dataset_indices:
            row = rows[int(dataset_idx)]
            file_idx = int(row[0])
            for center_time_idx in row[2:]:
                time_indices.append(self._time_lookup[(file_idx, int(center_time_idx))])
        return np.asarray(time_indices, dtype=np.int64)

    def _batch_patch_cells(self, batch: Any, batch_size: int) -> List[np.ndarray]:
        patch_index_zooms = batch[-1]
        if self.zoom in patch_index_zooms:
            patch_indices = _as_numpy_1d(patch_index_zooms[self.zoom])
        else:
            patch_indices = _as_numpy_1d(patch_index_zooms[str(self.zoom)])
        if patch_indices.size != batch_size:
            raise ValueError(
                f"Patch index count {patch_indices.size} does not match output batch size {batch_size}."
            )
        return [
            self.dataset.get_indices_from_patch_idx(self.zoom, int(patch_idx)).reshape(-1)
            for patch_idx in patch_indices
        ]

    def _batch_patch_cells_for_zoom(
        self,
        patch_index_zooms: Mapping[Any, Any],
        batch_size: int,
        zoom: int,
    ) -> List[np.ndarray]:
        if zoom in patch_index_zooms:
            patch_indices = _as_numpy_1d(patch_index_zooms[zoom])
        else:
            patch_indices = _as_numpy_1d(patch_index_zooms[str(zoom)])
        if patch_indices.size != batch_size:
            raise ValueError(
                f"Patch index count {patch_indices.size} does not match output batch size {batch_size}."
            )
        return [
            self.dataset.get_indices_from_patch_idx(zoom, int(patch_idx)).reshape(-1)
            for patch_idx in patch_indices
        ]

    def _write_variable(
        self,
        arrays: Dict[str, Any],
        variable: str,
        tensor: torch.Tensor,
        time_indices: np.ndarray,
        cells: List[np.ndarray],
    ) -> None:
        data = self.dataset.var_normalizers[self.normalizer_zoom][variable].denormalize(
            tensor.detach().cpu().float()
        )
        data_np = data.numpy().astype(np.float32, copy=False)

        for batch_pos, time_idx in enumerate(time_indices):
            cell_idx = cells[batch_pos]
            sample = data_np[batch_pos]
            if sample.shape[-1] == 1:
                sample = sample[..., 0]

            if sample.ndim == 3 and sample.shape[-1] == 1:
                arrays[variable].oindex[time_idx, :, cell_idx] = sample[:, :, 0]
            elif sample.ndim == 3:
                arrays[variable].oindex[time_idx, :, :, cell_idx] = np.moveaxis(sample, 1, 2)
            elif sample.ndim == 4:
                arrays[variable].oindex[time_idx, :, :, cell_idx, :] = np.moveaxis(sample, 1, 2)
            else:
                raise ValueError(f"Unsupported sample shape for {variable}: {sample.shape}")

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: Any,
        prediction: Mapping[str, Any],
        batch_indices: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if not trainer.is_global_zero:
            return
        if prediction is None or "output" not in prediction:
            return

        self._init_store(prediction)
        first_group = next(group for group in prediction["output"] if group)
        first_tensor = _get_zoom_tensor(first_group, self.zoom)
        batch_size = int(first_tensor.shape[0])
        time_indices = self._batch_time_indices(batch_indices)
        if time_indices.size != batch_size:
            raise ValueError(
                f"Time index count {time_indices.size} does not match output batch size {batch_size}."
            )
        cells = self._batch_patch_cells(batch, batch_size)

        for group_pred, variables in zip(prediction["output"], self.grouped_variables):
            if not group_pred:
                continue
            group_tensor = _get_zoom_tensor(group_pred, self.zoom)
            for local_var_idx, variable in enumerate(variables):
                self._write_variable(self._arrays, variable, group_tensor[:, local_var_idx], time_indices, cells)

        combined_first_group = next(group for group in prediction["output_combined"] if group)
        combined_batch_size = int(_get_zoom_tensor(combined_first_group, self.combined_zoom).shape[0])
        if combined_batch_size != batch_size:
            raise ValueError(
                f"Combined output batch size {combined_batch_size} does not match raw output batch size {batch_size}."
            )
        combined_cells = self._batch_patch_cells_for_zoom(
            patch_index_zooms=prediction["patch_index_zooms"],
            batch_size=combined_batch_size,
            zoom=self.combined_zoom,
        )
        for group_pred, variables in zip(prediction["output_combined"], self.grouped_variables):
            if not group_pred:
                continue
            group_tensor = _get_zoom_tensor(group_pred, self.combined_zoom)
            for local_var_idx, variable in enumerate(variables):
                self._write_variable(
                    self._combined_arrays,
                    variable,
                    group_tensor[:, local_var_idx],
                    time_indices,
                    combined_cells,
                )


@hydra.main(version_base=None, config_path="/Users/maxwitte/work/stableclimgen/fieldspacenn/configs", config_name="era5_prediction_rollout")
def rollout(cfg: DictConfig) -> None:
    test_dataset: BaseDataset = instantiate(cfg.dataloader.dataset, data_dict=cfg.data_split["test"])

    model: Any = instantiate(cfg.model)
    if cfg.ckpt_path is not None:
        model = load_from_state_dict(model, cfg.ckpt_path, print_keys=True, device=model.device)[0]

    data_module: DataModule = instantiate(cfg.dataloader.datamodule, dataset_test=test_dataset)

    zoom = int(cfg.rollout.zoom) if cfg.rollout.zoom is not None else max(model.model.in_zooms)
    writer = HealPixZarrPredictionWriter(
        output_path=cfg.rollout.output_path,
        dataset=test_dataset,
        zoom=zoom,
        n_autoregressive_steps=model.n_autoregressive_steps,
        ckpt_path=cfg.ckpt_path,
        overwrite=cfg.rollout.overwrite,
        time_chunk=cfg.rollout.time_chunk,
        prediction_timedelta_chunk=cfg.rollout.prediction_timedelta_chunk,
        convention=cfg.rollout.convention,
    )

    trainer: Trainer = instantiate(cfg.trainer)
    trainer.callbacks.append(writer)

    start_time = time.time()
    trainer.predict(
        model=model,
        dataloaders=data_module.test_dataloader(),
        return_predictions=False,
    )
    end_time = time.time()
    print(f"Wrote rollout to {cfg.rollout.output_path}")
    print(f"Rollout time: {end_time - start_time:.2f} seconds")
    print("Rollout config:")
    print(OmegaConf.to_yaml(cfg.rollout))


if __name__ == "__main__":
    rollout()
