"""Encode a dataset with a trained multi-grid autoencoder and save its latents."""

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
from omegaconf import ListConfig, OmegaConf, open_dict

from .utils.helpers import load_from_state_dict


def _cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu").contiguous()


def concatenate_latents(
    predictions: Sequence[Mapping[str, Any]],
) -> List[Dict[int, torch.Tensor]]:
    """Concatenate every predicted group and zoom along the batch dimension."""
    if not predictions:
        raise ValueError("The encoder returned no prediction batches.")

    outputs = [prediction.get("output") for prediction in predictions]
    if any(not isinstance(output, (list, tuple)) for output in outputs):
        raise TypeError("Expected each prediction's `output` to be a list of latent groups.")

    group_count = len(outputs[0])
    if any(len(output) != group_count for output in outputs):
        raise ValueError("The number of latent groups changed between prediction batches.")

    latents: List[Dict[int, torch.Tensor]] = []
    for group_index in range(group_count):
        group_batches = [output[group_index] for output in outputs]
        populated_groups = [group for group in group_batches if group]
        if not populated_groups:
            latents.append({})
            continue
        if any(not isinstance(group, Mapping) for group in populated_groups):
            raise TypeError(f"Latent group {group_index} is not a zoom-to-tensor mapping.")

        zooms = {int(zoom) for zoom in populated_groups[0]}
        for group in populated_groups[1:]:
            if {int(zoom) for zoom in group} != zooms:
                raise ValueError(
                    f"The available zooms changed between batches for latent group {group_index}."
                )

        group_latents: Dict[int, torch.Tensor] = {}
        for zoom in sorted(zooms):
            tensors = []
            for group in populated_groups:
                tensor = group.get(zoom, group.get(str(zoom)))
                if not torch.is_tensor(tensor):
                    raise TypeError(
                        f"Latent group {group_index}, zoom {zoom} is not a tensor."
                    )
                tensors.append(_cpu_tensor(tensor))
            group_latents[zoom] = torch.cat(tensors, dim=0)
        latents.append(group_latents)

    return latents


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


def _source_coordinates(dataset: Any, expected_times: int) -> Dict[str, Any]:
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
    time = np.asarray(time_values)
    if len(time) != expected_times:
        raise ValueError(
            "Source time-coordinate length does not match encoded batch dimension: "
            f"time={len(time)}, encoded={expected_times}."
        )

    return {
        "time": time,
        "time_attrs": time_attrs,
        "level": level_values,
        "level_attrs": level_attrs,
    }


def _create_array(
    group: Any,
    name: str,
    data: np.ndarray,
    dimensions: Sequence[str],
    chunks: Sequence[int] | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Any:
    chunk_shape = tuple(chunks or data.shape)
    kwargs = {
        "shape": tuple(data.shape),
        "chunks": chunk_shape,
        "dtype": data.dtype,
        "fill_value": np.nan if np.issubdtype(data.dtype, np.floating) else 0,
    }
    try:
        array = group.create_array(name, dimension_names=tuple(dimensions), **kwargs)
    except TypeError:
        array = group.create_array(name, **kwargs)
    array.attrs["_ARRAY_DIMENSIONS"] = list(dimensions)
    if attrs:
        array.attrs.update(dict(attrs))
    array[...] = data
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

    # Put the spatial cell after level so the conventional climate-field layout
    # becomes (time, level, cell) when all auxiliary latent axes are singleton.
    data = tensor[:, :, variable_index].permute(0, 1, 2, 4, 3, 5)
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
    return data[selectors].numpy(), retained_dimensions


def save_zarr(
    output_path: Path,
    latents: Sequence[Mapping[int, torch.Tensor]],
    dataset: Any,
    checkpoint: Path,
    config_dir: Path,
    config_name: str,
    test_split: Any,
    time_chunk: int,
    overwrite: bool,
) -> None:
    groups = _variable_groups(dataset)
    if len(groups) != len(latents):
        raise ValueError(
            "Latent group count does not match configured variable groups: "
            f"latents={len(latents)}, variables={len(groups)}."
        )
    if not latents or not any(latents):
        raise ValueError("No populated latent groups were produced.")

    first_tensor = next(tensor for group in latents for tensor in group.values())
    source_coordinates = _source_coordinates(dataset, expected_times=int(first_tensor.shape[0]))
    zooms = sorted({int(zoom) for group in latents for zoom in group})

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output store already exists: {output_path}. Pass --overwrite to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.", dir=str(output_path.parent))
    )

    try:
        root = zarr.open_group(str(temporary_path), mode="w")
        root.attrs.update(
            {
                "format_version": 1,
                "representation": "fieldspacenn_autoencoder_latent",
                "checkpoint": str(checkpoint),
                "config_dir": str(config_dir),
                "config_name": config_name,
                "variable_groups": groups,
                "test_split": OmegaConf.to_container(test_split, resolve=True),
                "zoom_groups": [f"zoom_{zoom}" for zoom in zooms],
            }
        )

        for zoom in zooms:
            zoom_group = root.create_group(f"zoom_{zoom}")
            zoom_group.attrs.update(
                {
                    "healpix_nested": True,
                    "healpix_zoom": zoom,
                    "representation": "autoencoder_latent",
                }
            )
            _create_array(
                zoom_group,
                "time",
                source_coordinates["time"],
                ("time",),
                chunks=(min(time_chunk, len(source_coordinates["time"])),),
                attrs=source_coordinates["time_attrs"],
            )

            cell_count = int(next(group[zoom].shape[4] for group in latents if zoom in group))
            expected_cell_count = 12 * (2**zoom) ** 2
            if cell_count != expected_cell_count:
                raise ValueError(
                    f"Zoom {zoom} has {cell_count} cells; expected {expected_cell_count} for HEALPix."
                )
            _create_array(
                zoom_group,
                "cell",
                np.arange(cell_count, dtype=np.int64),
                ("cell",),
            )

            coordinate_sizes: Dict[str, int] = {}
            for latent_group, variable_group in zip(latents, groups):
                if zoom not in latent_group:
                    continue
                tensor = latent_group[zoom]
                variables = variable_group["variables"]
                if int(tensor.shape[2]) != len(variables):
                    raise ValueError(
                        f"Group {variable_group['name']} at zoom {zoom} contains "
                        f"{tensor.shape[2]} variables, expected {len(variables)}: {variables}."
                    )
                for variable_index, variable in enumerate(variables):
                    data, dimensions = _variable_data_and_dimensions(tensor, variable_index)
                    chunks = list(data.shape)
                    chunks[0] = min(time_chunk, data.shape[0])
                    _create_array(
                        zoom_group,
                        variable,
                        data,
                        dimensions,
                        chunks=chunks,
                        attrs={
                            "representation": "autoencoder_latent",
                            "source_variable": variable,
                            "variable_group": variable_group["name"],
                            "coordinates": "time cell" + (" level" if "level" in dimensions else ""),
                        },
                    )
                    for dimension, size in zip(dimensions, data.shape):
                        coordinate_sizes[dimension] = int(size)

            if "level" in coordinate_sizes:
                level = source_coordinates["level"]
                if level is None:
                    overwrite_depths = getattr(dataset, "overwrite_depths", None)
                    level = (
                        np.arange(coordinate_sizes["level"], dtype=np.int64)
                        if overwrite_depths is None
                        else np.asarray(overwrite_depths.detach().cpu())
                    )
                if len(level) != coordinate_sizes["level"]:
                    raise ValueError(
                        "Source level coordinate does not match the latent level dimension: "
                        f"source={len(level)}, "
                        f"latent={coordinate_sizes['level']}."
                    )
                _create_array(
                    zoom_group,
                    "level",
                    level,
                    ("level",),
                    attrs=source_coordinates["level_attrs"],
                )
            for dimension in ("sample", "token_time", "feature"):
                if dimension in coordinate_sizes:
                    _create_array(
                        zoom_group,
                        dimension,
                        np.arange(coordinate_sizes[dimension], dtype=np.int64),
                        (dimension,),
                    )

        if output_path.exists():
            if output_path.is_dir():
                shutil.rmtree(output_path)
            else:
                output_path.unlink()
        os.replace(temporary_path, output_path)
    except Exception:
        shutil.rmtree(temporary_path, ignore_errors=True)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", required=True, help="Directory containing the Hydra config.")
    parser.add_argument("--config-name", default="composed_config", help="Hydra config name.")
    parser.add_argument("--checkpoint", required=True, help="Trained autoencoder checkpoint.")
    parser.add_argument("--output", required=True, help="Destination .zarr store.")
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
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=_parse_devices(args.devices),
        precision=args.precision,
        logger=False,
        enable_checkpointing=False,
    )

    predictions = trainer.predict(model=model, dataloaders=data_module.test_dataloader())
    latents = concatenate_latents(predictions)
    save_zarr(
        output_path=output_path,
        latents=latents,
        dataset=test_dataset,
        checkpoint=checkpoint,
        config_dir=config_dir,
        config_name=args.config_name,
        test_split=cfg.data_split.test,
        time_chunk=args.time_chunk,
        overwrite=args.overwrite,
    )

    print(f"Saved {len(test_dataset)} encoded sample(s) to {output_path}")
    for group_index, zooms in enumerate(latents):
        for zoom, tensor in zooms.items():
            print(f"  group={group_index} zoom={zoom} shape={tuple(tensor.shape)} dtype={tensor.dtype}")


if __name__ == "__main__":
    main()
