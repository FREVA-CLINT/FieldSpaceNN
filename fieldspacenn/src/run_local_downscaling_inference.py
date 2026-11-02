#!/usr/bin/env python3

import argparse
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import xarray as xr
from hydra.utils import instantiate
from lightning.pytorch.callbacks import Callback
from omegaconf import DictConfig, OmegaConf

from fieldspacenn.src.modules.grids.grid_utils import decode_zooms
from fieldspacenn.src.rollout import build_prediction_trainer
from fieldspacenn.src.utils.helpers import merge_sampling_dicts


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SNAPSHOT_DIR = (
    PROJECT_ROOT
    / "run/snapshots/stableclimgen/nextgems_hamburg_downscaling_local"
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run local downscaling inference from a composed config and Lightning "
            "checkpoint, then write the denormalized prediction as Zarr."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_SNAPSHOT_DIR / "composed_config.yaml",
        help="Path to the training run's composed_config.yaml.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_SNAPSHOT_DIR / "last.ckpt",
        help="Path to the pretrained Lightning checkpoint.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output Zarr path.")
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
        help="Configured data split to run.",
    )
    parser.add_argument(
        "--timesteps",
        default=None,
        help=(
            "Optional split timestep override. Accepts comma-separated indices and "
            "half-open ranges such as '1-5,10'. Use 'all' for every valid timestep."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--accelerator", choices=("auto", "cpu", "gpu"), default="auto"
    )
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument(
        "--input-zarr",
        type=Path,
        default=None,
        help=(
            "Optional replacement for the selected split's lowest-zoom source and "
            "target Zarr."
        ),
    )
    parser.add_argument(
        "--target-template-zarr",
        type=Path,
        default=None,
        help=(
            "Optional replacement for the selected split's highest-zoom source and "
            "target Zarr. Its spatial coordinates and variable metadata are copied."
        ),
    )
    parser.add_argument(
        "--norm-dict",
        type=Path,
        default=None,
        help="Optional normalization dictionary override.",
    )
    parser.add_argument(
        "--config-override",
        action="append",
        default=[],
        help="OmegaConf dot-list override; may be supplied more than once.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.num_workers is not None and args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.devices <= 0:
        parser.error("--devices must be positive")
    return args


def parse_timesteps(value: str) -> Optional[list[int]]:
    value = value.strip()
    if value.lower() == "all":
        return None
    if not value:
        raise ValueError("The timestep override cannot be empty.")

    timesteps: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            raise ValueError(f"Invalid empty timestep in {value!r}.")
        if "-" not in token:
            timesteps.append(int(token))
            continue

        parts = token.split("-")
        if len(parts) != 2 or not all(parts):
            raise ValueError(f"Invalid timestep range {token!r}.")
        start, end = map(int, parts)
        if end <= start:
            raise ValueError(
                f"Timestep range {token!r} must have an end greater than its start."
            )
        timesteps.extend(range(start, end))

    if len(set(timesteps)) != len(timesteps):
        raise ValueError("The timestep override contains duplicate indices.")
    return timesteps


def zoom_entry(mapping: Mapping[Any, Any], zoom: int) -> Any:
    if zoom in mapping:
        return mapping[zoom]
    if str(zoom) in mapping:
        return mapping[str(zoom)]
    raise KeyError(f"Zoom {zoom} is not configured; available keys: {list(mapping)}")


def configure_run(cfg: DictConfig, args: argparse.Namespace) -> DictConfig:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    if args.config_override:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.config_override))

    if args.split not in cfg.data_split:
        raise KeyError(f"Split {args.split!r} is not present in cfg.data_split.")
    split_cfg = cfg.data_split[args.split]

    if args.timesteps is not None:
        timesteps = parse_timesteps(args.timesteps)
        if timesteps is None:
            split_cfg.pop("timesteps", None)
        else:
            split_cfg.timesteps = timesteps

    source_zooms = sorted(int(zoom) for zoom in split_cfg.source.keys())
    target_zooms = sorted(int(zoom) for zoom in split_cfg.target.keys())
    if not source_zooms or not target_zooms:
        raise ValueError("The selected split must configure source and target zooms.")

    if args.input_zarr is not None:
        input_path = str(args.input_zarr.resolve())
        lowest_zoom = min(source_zooms + target_zooms)
        zoom_entry(split_cfg.source, lowest_zoom).files = [input_path]
        zoom_entry(split_cfg.target, lowest_zoom).files = [input_path]
    if args.target_template_zarr is not None:
        template_path = str(args.target_template_zarr.resolve())
        highest_zoom = max(source_zooms + target_zooms)
        zoom_entry(split_cfg.source, highest_zoom).files = [template_path]
        zoom_entry(split_cfg.target, highest_zoom).files = [template_path]
    if args.norm_dict is not None:
        cfg.dataloader.dataset.norm_dict = str(args.norm_dict.resolve())

    if args.batch_size is not None:
        cfg.dataloader.datamodule.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.dataloader.datamodule.num_workers = args.num_workers
        cfg.dataloader.datamodule.num_val_workers = args.num_workers
    cfg.dataloader.datamodule.shuffle = False
    cfg.dataloader.datamodule.persistent_workers = False

    cfg.trainer.accelerator = args.accelerator
    OmegaConf.update(cfg, "trainer.devices", args.devices, merge=False, force_add=True)
    return cfg


def grouped_variables(data_dict: Mapping[str, Any]) -> list[list[str]]:
    groups = []
    for group_name, variables in data_dict["variables"].items():
        if group_name in ("embedding", "embedding_1D"):
            continue
        groups.append(list(variables.keys()) if isinstance(variables, Mapping) else list(variables))
    return groups


def load_checkpoint_strict(model: torch.nn.Module, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain a state_dict.")

    model_state = model.state_dict()
    checkpoint_state = checkpoint["state_dict"]
    missing = sorted(set(model_state) - set(checkpoint_state))
    unexpected = sorted(set(checkpoint_state) - set(model_state))
    shape_mismatches = sorted(
        key
        for key in set(model_state) & set(checkpoint_state)
        if model_state[key].shape != checkpoint_state[key].shape
    )
    if missing or unexpected or shape_mismatches:
        details = []
        if missing:
            details.append(f"missing keys: {missing}")
        if unexpected:
            details.append(f"unexpected keys: {unexpected}")
        if shape_mismatches:
            details.append(f"shape mismatches: {shape_mismatches}")
        raise ValueError(
            "The composed config is incompatible with the checkpoint; refusing to "
            "run with partially loaded pretrained weights (" + "; ".join(details) + ")."
        )
    model.load_state_dict(checkpoint_state, strict=True)


def prediction_to_surface_array(tensor: torch.Tensor, variable: str) -> np.ndarray:
    # After selecting the variable axis, local 2-D fields have
    # (batch, internal_time, cell, depth, feature) layout.
    if tensor.ndim != 5:
        raise ValueError(
            f"Expected a 5-D tensor for {variable}, got shape {tuple(tensor.shape)}."
        )
    singleton_axes = (1, 3, 4)
    if any(tensor.shape[axis] != 1 for axis in singleton_axes):
        raise ValueError(
            f"Expected singleton time/depth/feature axes for {variable}, got "
            f"shape {tuple(tensor.shape)}."
        )
    return tensor[:, 0, :, 0, 0].detach().cpu().numpy().astype(np.float32, copy=False)


class LocalPredictionCollector(Callback):
    def __init__(self, dataset: Any) -> None:
        super().__init__()
        self.dataset = dataset
        self.variable_groups = grouped_variables(dataset.data_dict)
        self.output_zoom = max(int(zoom) for zoom in dataset.sampling_zooms_target)
        self.normalizer_zoom = max(int(zoom) for zoom in dataset.var_normalizers)
        self.values = {
            variable: [] for variables in self.variable_groups for variable in variables
        }

    def on_predict_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not isinstance(outputs, Mapping) or outputs.get("output") is None:
            raise ValueError(f"Prediction batch {batch_idx} has no output groups.")

        output_groups = outputs["output"]
        if len(output_groups) != len(self.variable_groups):
            raise ValueError(
                "Prediction group count does not match configured variable groups: "
                f"{len(output_groups)} != {len(self.variable_groups)}."
            )

        sample_configs = self.dataset.sampling_zooms_collate or self.dataset.sampling_zooms
        target_configs = self.dataset.sampling_zooms_target or sample_configs
        target_configs = merge_sampling_dicts(target_configs, batch[-1])

        for variables, output_group in zip(self.variable_groups, output_groups):
            if not output_group:
                raise ValueError(f"Prediction batch {batch_idx} contains an empty group.")
            decoded = decode_zooms(
                output_group.copy(),
                sample_configs=target_configs,
                out_zoom=self.output_zoom,
            )[self.output_zoom]
            if decoded.shape[1] != len(variables):
                raise ValueError(
                    f"Prediction has {decoded.shape[1]} variables, expected {len(variables)}."
                )
            for variable_index, variable in enumerate(variables):
                values = self.dataset.var_normalizers[self.normalizer_zoom][
                    variable
                ].denormalize(decoded[:, variable_index])
                self.values[variable].append(
                    prediction_to_surface_array(values, variable)
                )

    def arrays(self) -> dict[str, np.ndarray]:
        missing = [name for name, batches in self.values.items() if not batches]
        if missing:
            raise ValueError(f"No predictions were collected for variables: {missing}")
        return {name: np.concatenate(batches, axis=0) for name, batches in self.values.items()}


def configured_files(data_dict: Mapping[str, Any], section: str, zoom: int) -> list[str]:
    files = zoom_entry(data_dict[section], zoom)["files"]
    return [str(path) for path in files]


def output_coordinates(dataset: Any, output_zoom: int) -> tuple[np.ndarray, np.ndarray, Path]:
    rows = np.asarray(dataset.index_map[output_zoom])
    if rows.ndim != 2 or rows.shape[1] != 3:
        raise ValueError(
            "Local inference expects one region and one center timestep per item; "
            f"index map shape is {rows.shape}."
        )
    if len(np.unique(rows[:, :3], axis=0)) != len(rows):
        raise ValueError("The selected dataset contains duplicate output samples.")

    template_files = configured_files(dataset.data_dict, "target", output_zoom)
    time_by_file: dict[int, np.ndarray] = {}
    for file_index in np.unique(rows[:, 0]):
        path = template_files[int(file_index)]
        with xr.open_dataset(path) as source:
            time_by_file[int(file_index)] = np.asarray(source.time.values)

    target_shift = int(getattr(dataset, "target_time_shift", 0))
    time_values = np.asarray(
        [
            time_by_file[int(file_index)][int(time_index) + target_shift]
            for file_index, _, time_index in rows
        ]
    )
    return rows[:, 2].astype(np.int64), time_values, Path(template_files[0])


def build_output_dataset(
    arrays: Mapping[str, np.ndarray],
    dataset: Any,
    checkpoint_path: Path,
    config_path: Path,
    split: str,
) -> xr.Dataset:
    output_zoom = max(int(zoom) for zoom in dataset.sampling_zooms_target)
    timestep_indices, time_values, template_path = output_coordinates(
        dataset, output_zoom
    )
    expected_samples = len(time_values)
    for variable, values in arrays.items():
        if values.shape[0] != expected_samples:
            raise ValueError(
                f"{variable} has {values.shape[0]} samples, expected {expected_samples}."
            )

    with xr.open_dataset(template_path) as template:
        cell = template.cell.load()
        if any(values.shape[1] != cell.size for values in arrays.values()):
            raise ValueError("Prediction cell count does not match the target template.")

        result = xr.Dataset(
            data_vars={
                variable: xr.DataArray(
                    values,
                    dims=("time", "cell"),
                    attrs=dict(template[variable].attrs),
                )
                for variable, values in arrays.items()
            },
            coords={"time": time_values, "cell": cell.values},
            attrs={
                "checkpoint": str(checkpoint_path.resolve()),
                "composed_config": str(config_path.resolve()),
                "data_split": split,
                "downscaling_source_zoom": min(int(zoom) for zoom in dataset.sampling_zooms),
                "downscaling_target_zoom": output_zoom,
            },
        )
        result.time.attrs = dict(template.time.attrs)
        result.cell.attrs = dict(template.cell.attrs)
        result = result.assign_coords(
            source_timestep_index=("time", timestep_indices)
        )
        if "crs" in template:
            result = result.assign_coords(crs=template.crs.load())
    # Template arrays can retain backend-specific codecs that are invalid when
    # creating a fresh store with a different Zarr format.
    for variable in result.variables.values():
        variable.encoding.clear()
    return result


def write_zarr(result: xr.Dataset, output_path: Path, overwrite: bool) -> None:
    output_path = output_path.resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Pass --overwrite to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    if temporary_path.exists():
        shutil.rmtree(temporary_path)

    time_chunk = min(128, result.sizes["time"])
    encoding = {
        variable: {"chunks": (time_chunk, result.sizes["cell"])}
        for variable in result.data_vars
    }
    try:
        result.to_zarr(
            temporary_path,
            mode="w",
            consolidated=True,
            encoding=encoding,
        )
        if output_path.exists():
            shutil.rmtree(output_path)
        temporary_path.replace(output_path)
    except Exception:
        if temporary_path.exists():
            shutil.rmtree(temporary_path)
        raise


def run(args: argparse.Namespace) -> xr.Dataset:
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Composed config does not exist: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    cfg = configure_run(OmegaConf.load(config_path), args)
    dataset = instantiate(cfg.dataloader.dataset, data_dict=cfg.data_split[args.split])
    datamodule = instantiate(cfg.dataloader.datamodule, dataset_test=dataset)

    model = instantiate(cfg.model)
    load_checkpoint_strict(model, checkpoint_path)
    model.eval()

    collector = LocalPredictionCollector(dataset)
    trainer = build_prediction_trainer(cfg.trainer, run_name="local_downscaling")
    trainer.callbacks.append(collector)
    trainer.predict(
        model=model,
        dataloaders=datamodule.test_dataloader(),
        return_predictions=False,
    )

    result = build_output_dataset(
        collector.arrays(),
        dataset,
        checkpoint_path,
        config_path,
        args.split,
    )
    write_zarr(result, args.output, args.overwrite)
    return result


def main() -> None:
    args = parse_args()
    result = run(args)
    print(f"Wrote {args.output.resolve()}")
    print(result)


if __name__ == "__main__":
    main()
