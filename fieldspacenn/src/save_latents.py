"""Encode a dataset into one Zarr store per latent zoom level."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch
import zarr
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
from omegaconf import ListConfig, OmegaConf, open_dict

from .utils.helpers import load_from_state_dict


def _variable_groups(dataset: Any) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    for name, definitions in dataset.data_dict["variables"].items():
        if name in ("embedding", "embedding_1D"):
            continue
        groups.append(
            {
                "name": str(name),
                "variables": [str(variable) for variable in definitions],
            }
        )
    return groups


def _mapping_value(mapping: Mapping[Any, Any], key: int) -> Any:
    if key in mapping:
        return mapping[key]
    if str(key) in mapping:
        return mapping[str(key)]
    raise KeyError(f"Key {key} not found in {list(mapping)}")


def _zoom_tensor(mapping: Mapping[Any, torch.Tensor], zoom: int) -> torch.Tensor:
    tensor = mapping.get(zoom, mapping.get(str(zoom)))
    if not torch.is_tensor(tensor):
        raise TypeError(f"Latent zoom {zoom} is not a tensor.")
    return tensor


def _as_file_list(files: Any) -> List[str]:
    if isinstance(files, ListConfig):
        return [str(path) for path in OmegaConf.to_container(files, resolve=True)]
    if isinstance(files, (list, tuple)):
        return [str(path) for path in files]
    return [str(files)]


def _source_files(dataset: Any, zoom: int) -> List[str]:
    sources = dataset.data_dict["source"]
    if zoom not in sources and str(zoom) not in sources:
        zoom = max(int(source_zoom) for source_zoom in sources)
    return _as_file_list(_mapping_value(sources, zoom)["files"])


def _source_coordinates(dataset: Any) -> Dict[str, Any]:
    reference_zoom = min(int(zoom) for zoom in dataset.index_map)
    files = _source_files(dataset, reference_zoom)
    time_by_file: Dict[int, np.ndarray] = {}
    time_attrs: Dict[str, Any] = {}
    level_values: np.ndarray | None = None
    level_attrs: Dict[str, Any] = {}

    for file_index, source_file in enumerate(files):
        source = zarr.open_group(source_file, mode="r")
        if "time" not in source:
            raise KeyError(f"Source store has no `time` coordinate: {source_file}")
        time_by_file[file_index] = np.asarray(source["time"][:])
        if not time_attrs:
            time_attrs = dict(source["time"].attrs)
        if level_values is None and "level" in source:
            level_values = np.asarray(source["level"][:])
            level_attrs = dict(source["level"].attrs)

    time_values = []
    for row in dataset.index_map[reference_zoom]:
        file_index = int(row[0])
        for source_time_index in row[2:]:
            time_values.append(time_by_file[file_index][int(source_time_index)])

    return {
        "reference_zoom": reference_zoom,
        "time": np.asarray(time_values),
        "time_attrs": time_attrs,
        "level": level_values,
        "level_attrs": level_attrs,
    }


def _flatten_batch_indices(batch_indices: Any) -> List[int]:
    if batch_indices is None:
        return []
    if torch.is_tensor(batch_indices):
        return [int(index) for index in batch_indices.detach().cpu().view(-1).tolist()]
    if isinstance(batch_indices, np.ndarray):
        return [int(index) for index in batch_indices.reshape(-1).tolist()]
    if isinstance(batch_indices, (list, tuple)):
        indices: List[int] = []
        for item in batch_indices:
            indices.extend(_flatten_batch_indices(item))
        return indices
    return [int(batch_indices)]


def _create_empty_array(
    group: Any,
    name: str,
    shape: Sequence[int],
    chunks: Sequence[int],
    dtype: Any,
    dimensions: Sequence[str],
    attrs: Mapping[str, Any] | None = None,
    fill_value: Any = None,
) -> Any:
    if fill_value is None:
        fill_value = np.nan if np.issubdtype(np.dtype(dtype), np.floating) else 0
    kwargs = {
        "shape": tuple(shape),
        "chunks": tuple(chunks),
        "dtype": dtype,
        "fill_value": fill_value,
    }
    try:
        array = group.create_array(name, dimension_names=tuple(dimensions), **kwargs)
    except TypeError:
        try:
            array = group.create_array(name, **kwargs)
        except AttributeError:
            array = group.create_dataset(name, **kwargs)
    except AttributeError:
        array = group.create_dataset(name, **kwargs)

    array.attrs["_ARRAY_DIMENSIONS"] = list(dimensions)
    if attrs:
        array.attrs.update(dict(attrs))
    return array


def _create_coordinate(
    group: Any,
    name: str,
    values: np.ndarray,
    attrs: Mapping[str, Any] | None = None,
    chunk_size: int | None = None,
) -> Any:
    values = np.asarray(values)
    chunks = (min(chunk_size or len(values), len(values)),)
    array = _create_empty_array(
        group=group,
        name=name,
        shape=values.shape,
        chunks=chunks,
        dtype=values.dtype,
        dimensions=(name,),
        attrs=attrs,
    )
    array[...] = values
    return array


def _variable_data_and_dimensions(
    tensor: torch.Tensor,
    variable_index: int,
) -> tuple[np.ndarray, List[str]]:
    if tensor.ndim != 7:
        raise ValueError(
            "Expected latent tensors with layout "
            "(batch, sample, variable, token_time, cell, level, feature), "
            f"got {tuple(tensor.shape)}."
        )

    # Move only one variable from the accelerator to CPU at a time. Put cell
    # after level so the common layouts become (time, cell) and
    # (time, level, cell).
    data = (
        tensor[:, :, variable_index]
        .detach()
        .to(device="cpu")
        .permute(0, 1, 2, 4, 3, 5)
    )
    dimensions = ["time", "sample", "token_time", "level", "cell", "feature"]
    required_dimensions = {"time", "cell"}
    selectors = tuple(
        slice(None) if size > 1 or dimension in required_dimensions else 0
        for size, dimension in zip(data.shape, dimensions)
    )
    retained_dimensions = [
        dimension
        for size, dimension in zip(data.shape, dimensions)
        if size > 1 or dimension in required_dimensions
    ]
    return data[selectors].contiguous().numpy(), retained_dimensions


class LatentZarrPredictionWriter(BasePredictionWriter):
    """Write each predicted latent batch directly to its final Zarr region."""

    def __init__(
        self,
        output_path: Path,
        dataset: Any,
        checkpoint: Path,
        config_dir: Path,
        config_name: str,
        test_split: Any,
        time_chunk: int,
        overwrite: bool,
    ) -> None:
        super().__init__(write_interval="batch")
        self.output_path = output_path
        self.dataset = dataset
        self.checkpoint = checkpoint
        self.config_dir = config_dir
        self.config_name = config_name
        self.time_chunk = int(time_chunk)
        self.overwrite = bool(overwrite)
        self.variable_groups = _variable_groups(dataset)
        self.source_coordinates = _source_coordinates(dataset)
        self.n_times = len(self.source_coordinates["time"])
        self.reference_zoom = int(self.source_coordinates["reference_zoom"])

        if self.n_times == 0:
            raise ValueError("The configured dataset contains no source time coordinates.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary_path = Path(
            tempfile.mkdtemp(prefix=f".{output_path.name}.", dir=str(output_path.parent))
        )
        self.test_split = OmegaConf.to_container(test_split, resolve=True)
        self.zoom_stores: Dict[int, Any] = {}
        self.output_paths: Dict[int, Path] = {}
        self.arrays: Dict[tuple[int, str], Any] = {}
        self.coordinate_sizes: Dict[int, Dict[str, int]] = {}
        self.written_times: Dict[int, np.ndarray] = {}
        self.finalized = False

    def _zoom_output_path(self, zoom: int) -> Path:
        return self.output_path.with_name(
            f"{self.output_path.stem}_zoom_{zoom}{self.output_path.suffix}"
        )

    def _batch_time_indices(self, batch_indices: Any) -> np.ndarray:
        dataset_indices = _flatten_batch_indices(batch_indices)
        if not dataset_indices:
            raise ValueError("Lightning did not provide batch indices for streaming Zarr output.")

        rows = self.dataset.index_map[self.reference_zoom]
        time_indices: List[int] = []
        running_index = 0
        lookup: Dict[tuple[int, int], int] = {}
        for row in rows:
            file_index = int(row[0])
            for source_time_index in row[2:]:
                lookup[(file_index, int(source_time_index))] = running_index
                running_index += 1
        for dataset_index in dataset_indices:
            row = rows[dataset_index]
            file_index = int(row[0])
            for source_time_index in row[2:]:
                time_indices.append(lookup[(file_index, int(source_time_index))])
        return np.asarray(time_indices, dtype=np.int64)

    def _ensure_zoom_store(self, zoom: int, tensor: torch.Tensor) -> Any:
        if zoom in self.zoom_stores:
            return self.zoom_stores[zoom]

        cell_count = int(tensor.shape[4])
        expected_cell_count = 12 * (2**zoom) ** 2
        if cell_count != expected_cell_count:
            raise ValueError(
                f"Zoom {zoom} has {cell_count} cells; expected {expected_cell_count} for HEALPix. "
                "Streaming output currently requires full-globe latent batches."
            )

        output_path = self._zoom_output_path(zoom)
        if output_path.exists() and not self.overwrite:
            raise FileExistsError(
                f"Output store already exists: {output_path}. Pass --overwrite to replace it."
            )

        store_path = self.temporary_path / output_path.name
        store = zarr.open_group(str(store_path), mode="w")
        store.attrs.update(
            {
                "format_version": 1,
                "representation": "fieldspacenn_autoencoder_latent",
                "checkpoint": str(self.checkpoint),
                "config_dir": str(self.config_dir),
                "config_name": self.config_name,
                "variable_groups": self.variable_groups,
                "test_split": self.test_split,
                "healpix_nested": True,
                "healpix_zoom": zoom,
            }
        )
        _create_coordinate(
            store,
            "time",
            self.source_coordinates["time"],
            attrs=self.source_coordinates["time_attrs"],
            chunk_size=self.time_chunk,
        )
        _create_coordinate(store, "cell", np.arange(cell_count, dtype=np.int64))

        self.zoom_stores[zoom] = store
        self.output_paths[zoom] = output_path
        self.coordinate_sizes[zoom] = {"time": self.n_times, "cell": cell_count}
        self.written_times[zoom] = np.zeros(self.n_times, dtype=bool)
        return store

    def _ensure_dimension_coordinate(self, zoom: int, dimension: str, size: int) -> None:
        existing_size = self.coordinate_sizes[zoom].get(dimension)
        if existing_size is not None:
            if existing_size != size:
                raise ValueError(
                    f"Dimension {dimension} at zoom {zoom} changed from {existing_size} to {size}."
                )
            return

        store = self.zoom_stores[zoom]
        if dimension == "level":
            values = self.source_coordinates["level"]
            if values is None:
                overwrite_depths = getattr(self.dataset, "overwrite_depths", None)
                values = (
                    np.arange(size, dtype=np.int64)
                    if overwrite_depths is None
                    else np.asarray(overwrite_depths.detach().cpu())
                )
            if len(values) != size:
                raise ValueError(
                    "Source level coordinate does not match the latent level dimension: "
                    f"source={len(values)}, latent={size}."
                )
            attrs = self.source_coordinates["level_attrs"]
        else:
            values = np.arange(size, dtype=np.int64)
            attrs = None
        _create_coordinate(store, dimension, values, attrs=attrs)
        self.coordinate_sizes[zoom][dimension] = size

    def _ensure_variable_array(
        self,
        zoom: int,
        variable: str,
        variable_group: str,
        data: np.ndarray,
        dimensions: Sequence[str],
    ) -> Any:
        key = (zoom, variable)
        if key in self.arrays:
            array = self.arrays[key]
            expected_shape = (self.n_times, *data.shape[1:])
            if tuple(array.shape) != expected_shape:
                raise ValueError(
                    f"Latent shape for {variable} at zoom {zoom} changed: "
                    f"stored={array.shape}, batch={data.shape}."
                )
            return array

        for dimension, size in zip(dimensions[1:], data.shape[1:]):
            self._ensure_dimension_coordinate(zoom, dimension, int(size))

        shape = (self.n_times, *data.shape[1:])
        chunks = (min(self.time_chunk, self.n_times), *data.shape[1:])
        array = _create_empty_array(
            group=self.zoom_stores[zoom],
            name=variable,
            shape=shape,
            chunks=chunks,
            dtype=data.dtype,
            dimensions=dimensions,
            attrs={
                "representation": "autoencoder_latent",
                "source_variable": variable,
                "variable_group": variable_group,
                "coordinates": "time cell" + (" level" if "level" in dimensions else ""),
            },
        )
        self.arrays[key] = array
        return array

    @staticmethod
    def _write_time_rows(array: Any, time_indices: np.ndarray, data: np.ndarray) -> None:
        if len(np.unique(time_indices)) != len(time_indices):
            raise ValueError(f"A prediction batch contains duplicate time indices: {time_indices}.")
        if len(time_indices) > 0 and np.array_equal(
            time_indices, np.arange(time_indices[0], time_indices[0] + len(time_indices))
        ):
            array[int(time_indices[0]) : int(time_indices[-1]) + 1] = data
            return
        selection = (time_indices,) + (slice(None),) * (data.ndim - 1)
        array.oindex[selection] = data

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
            raise ValueError(f"Prediction batch {batch_idx} has no `output` entry.")
        output_groups = prediction["output"]
        if not isinstance(output_groups, (list, tuple)):
            raise TypeError("Prediction `output` must be a list of latent variable groups.")
        if len(output_groups) != len(self.variable_groups):
            raise ValueError(
                "Latent group count does not match configured variable groups: "
                f"latents={len(output_groups)}, variables={len(self.variable_groups)}."
            )

        first_group = next((group for group in output_groups if group), None)
        if first_group is None:
            raise ValueError(f"Prediction batch {batch_idx} contains no latent tensors.")
        first_zoom = int(next(iter(first_group)))
        batch_size = int(_zoom_tensor(first_group, first_zoom).shape[0])
        time_indices = self._batch_time_indices(batch_indices)
        if len(time_indices) != batch_size:
            raise ValueError(
                f"Time index count {len(time_indices)} does not match latent batch size {batch_size}."
            )

        for output_group, variable_group in zip(output_groups, self.variable_groups):
            if not output_group:
                continue
            variables = variable_group["variables"]
            for zoom_key in output_group:
                zoom = int(zoom_key)
                tensor = _zoom_tensor(output_group, zoom)
                if int(tensor.shape[0]) != batch_size:
                    raise ValueError(f"Batch size differs between latent groups at zoom {zoom}.")
                if int(tensor.shape[2]) != len(variables):
                    raise ValueError(
                        f"Group {variable_group['name']} at zoom {zoom} contains "
                        f"{tensor.shape[2]} variables, expected {len(variables)}: {variables}."
                    )
                self._ensure_zoom_store(zoom, tensor)
                for variable_index, variable in enumerate(variables):
                    data, dimensions = _variable_data_and_dimensions(tensor, variable_index)
                    array = self._ensure_variable_array(
                        zoom=zoom,
                        variable=variable,
                        variable_group=variable_group["name"],
                        data=data,
                        dimensions=dimensions,
                    )
                    self._write_time_rows(array, time_indices, data)

                self.written_times[zoom][time_indices] = True

    def finalize(self) -> None:
        if self.finalized:
            return
        if not self.arrays:
            raise ValueError("Streaming prediction created no latent variable arrays.")

        expected_variables = {
            variable
            for group in self.variable_groups
            for variable in group["variables"]
        }
        for zoom in self.zoom_stores:
            missing_times = np.flatnonzero(~self.written_times[zoom])
            if len(missing_times):
                preview = missing_times[:10].tolist()
                raise ValueError(
                    f"Zoom {zoom} did not write {len(missing_times)} time rows; "
                    f"first missing indices: {preview}."
                )
            stored_variables = {
                variable for stored_zoom, variable in self.arrays if stored_zoom == zoom
            }
            if stored_variables != expected_variables:
                raise ValueError(
                    f"Zoom {zoom} has incomplete variables: stored={sorted(stored_variables)}, "
                    f"expected={sorted(expected_variables)}."
                )

        for zoom, output_path in sorted(self.output_paths.items()):
            temporary_store = self.temporary_path / output_path.name
            zarr.consolidate_metadata(str(temporary_store))
            if output_path.exists():
                if output_path.is_dir():
                    shutil.rmtree(output_path)
                else:
                    output_path.unlink()
            os.replace(temporary_store, output_path)
        self.temporary_path.rmdir()
        self.finalized = True

    def abort(self) -> None:
        if not self.finalized:
            shutil.rmtree(self.temporary_path, ignore_errors=True)

    def summary_lines(self) -> List[str]:
        return [
            f"  store={self.output_paths[zoom]} variable={variable} "
            f"shape={tuple(array.shape)} dtype={array.dtype}"
            for (zoom, variable), array in sorted(self.arrays.items())
        ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", required=True, help="Directory containing the Hydra config.")
    parser.add_argument("--config-name", default="composed_config", help="Hydra config name.")
    parser.add_argument("--checkpoint", required=True, help="Trained autoencoder checkpoint.")
    parser.add_argument(
        "--output",
        required=True,
        help="Output name template; e.g. latents.zarr creates latents_zoom_<level>.zarr.",
    )
    parser.add_argument("--accelerator", default="cpu", help="Lightning accelerator (cpu, gpu, mps).")
    parser.add_argument("--devices", default="1", help="Lightning device count or device list.")
    parser.add_argument("--precision", default="32-true", help="Lightning inference precision.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--time-chunk", type=int, default=1, help="Zarr time chunk size.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output store.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Hydra override; repeat this option to provide more than one.",
    )
    return parser.parse_args()


def _parse_devices(value: str) -> Any:
    if value.isdigit():
        return int(value)
    if "," in value:
        return [int(device) for device in value.split(",")]
    return value


def _validate_single_device(devices: Any) -> None:
    if devices == 1:
        return
    if isinstance(devices, list) and len(devices) == 1:
        return
    raise ValueError(
        "Streaming latent export currently supports exactly one device because each rank "
        "would otherwise write only its own dataset shard. Use --devices 1."
    )


def main() -> None:
    args = _parse_args()
    config_dir = Path(args.config_dir).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not config_dir.is_dir():
        raise FileNotFoundError(f"Config directory does not exist: {config_dir}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    if output_path.suffix != ".zarr":
        raise ValueError(f"Output must use the .zarr extension: {output_path}")
    if args.time_chunk < 1:
        raise ValueError("--time-chunk must be at least 1.")
    devices = _parse_devices(args.devices)
    _validate_single_device(devices)

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=args.config_name, overrides=args.override)

    with open_dict(cfg):
        cfg.model.mode = "encode"
        cfg.model.n_samples = 1

    test_dataset = instantiate(cfg.dataloader.dataset, data_dict=cfg.data_split.test)
    if len(test_dataset) == 0:
        raise ValueError(
            "The configured test split is empty. Check `data_split.test.timesteps`; "
            "a range such as `1-2` contains one sample."
        )

    model = instantiate(cfg.model)
    model, _ = load_from_state_dict(
        model,
        str(checkpoint),
        device=torch.device("cpu"),
        print_keys=True,
    )
    data_module = instantiate(
        cfg.dataloader.datamodule,
        dataset_test=test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    writer = LatentZarrPredictionWriter(
        output_path=output_path,
        dataset=test_dataset,
        checkpoint=checkpoint,
        config_dir=config_dir,
        config_name=args.config_name,
        test_split=cfg.data_split.test,
        time_chunk=args.time_chunk,
        overwrite=args.overwrite,
    )
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=devices,
        precision=args.precision,
        logger=False,
        enable_checkpointing=False,
        callbacks=[writer],
    )

    try:
        trainer.predict(
            model=model,
            dataloaders=data_module.test_dataloader(),
            return_predictions=False,
        )
        writer.finalize()
    except BaseException:
        writer.abort()
        raise

    print(
        f"Streamed {writer.n_times} encoded sample(s) to "
        f"{len(writer.output_paths)} zoom-specific store(s)"
    )
    for line in writer.summary_lines():
        print(line)


if __name__ == "__main__":
    main()
