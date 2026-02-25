import os
import time
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional, Tuple

import healpy as hp
import hydra
import torch
from einops import rearrange
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from omegaconf import DictConfig

from .data.datasets_base import BaseDataset
from .data.pl_data_module import DataModule
from .utils.helpers import load_from_state_dict


def _get_zoom_tensor(zoom_dict: Mapping[Any, torch.Tensor], zoom: int) -> torch.Tensor:
    if zoom in zoom_dict:
        return zoom_dict[zoom]
    zoom_str = str(zoom)
    if zoom_str in zoom_dict:
        return zoom_dict[zoom_str]
    raise KeyError(f"Zoom {zoom} not found. Available keys: {list(zoom_dict.keys())}")


def _merge_patch_dimension(tensor: torch.Tensor, n_patches: int, var_axis: int) -> torch.Tensor:
    if var_axis == 1:
        return rearrange(tensor, "(b2 b1) v t n ... -> b2 v t (b1 n) ...", b1=n_patches)
    if var_axis == 2:
        return rearrange(tensor, "(b2 b1) m v t n ... -> b2 v m t (b1 n) ...", b1=n_patches)
    raise ValueError(f"Unsupported variable axis `{var_axis}`. Expected 1 or 2.")


def _group_and_variable_meta(data_dict: Mapping[str, Any]) -> Tuple[List[List[str]], List[str]]:
    grouped_variables: List[List[str]] = []
    for group_name, variables in data_dict["variables"].items():
        if group_name == "embedding":
            continue
        grouped_variables.append(list(variables))

    flat_variables: List[str] = [var for variables in grouped_variables for var in variables]
    duplicates = sorted([v for v in set(flat_variables) if flat_variables.count(v) > 1])
    if duplicates:
        raise ValueError(
            "Duplicate variable names across groups are not supported for output dictionary keys: "
            f"{duplicates}"
        )

    return grouped_variables, flat_variables


def _concat_prediction_groups(
    predictions: Sequence[Mapping[str, Any]],
    pred_key: str,
    zoom: int,
    expected_group_sizes: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, List[int], int]:
    batch_tensors: List[torch.Tensor] = []
    group_sizes: Optional[List[int]] = None
    var_axis: Optional[int] = None

    for batch_idx, batch in enumerate(predictions):
        if pred_key not in batch or batch[pred_key] is None:
            continue

        pred_value = batch[pred_key]
        if isinstance(pred_value, list):
            group_tensors: List[torch.Tensor] = []
            group_sizes_batch: List[int] = []
            expected_sizes_batch: List[int] = []

            for group_idx, group_pred in enumerate(pred_value):
                if not group_pred:
                    continue

                if isinstance(group_pred, Mapping):
                    group_tensor = _get_zoom_tensor(group_pred, zoom)
                elif torch.is_tensor(group_pred):
                    group_tensor = group_pred
                else:
                    raise TypeError(
                        f"Unsupported `{pred_key}` group type `{type(group_pred)}` in batch {batch_idx}."
                    )

                if expected_group_sizes is not None:
                    if group_idx >= len(expected_group_sizes):
                        raise ValueError(
                            f"Prediction returned more groups than expected for `{pred_key}`. "
                            f"group_idx={group_idx}, expected_groups={len(expected_group_sizes)}"
                        )
                    expected_sizes_batch.append(int(expected_group_sizes[group_idx]))
                group_tensors.append(group_tensor)

            if not group_tensors:
                continue

            if expected_sizes_batch:
                valid_axes = []
                for axis in (1, 2):
                    if all(tensor.dim() > axis and int(tensor.shape[axis]) == expected_size
                           for tensor, expected_size in zip(group_tensors, expected_sizes_batch)):
                        valid_axes.append(axis)

                if not valid_axes:
                    raise ValueError(
                        f"Could not infer variable axis for `{pred_key}` in batch {batch_idx}. "
                        f"Expected group sizes: {expected_sizes_batch}, "
                        f"tensor shapes: {[tuple(t.shape) for t in group_tensors]}"
                    )

                if var_axis is not None and var_axis in valid_axes:
                    var_dim = var_axis
                else:
                    var_dim = valid_axes[0]
            else:
                var_dim = 1 if group_tensors[0].dim() == 5 else 2

            batch_tensor = torch.cat(group_tensors, dim=var_dim)
            var_axis = var_dim
            group_sizes_batch = [int(group_tensor.shape[var_dim]) for group_tensor in group_tensors]
            if group_sizes is None:
                group_sizes = group_sizes_batch
            elif group_sizes != group_sizes_batch:
                raise ValueError(
                    f"Inconsistent group variable sizes for `{pred_key}` across batches: "
                    f"{group_sizes} vs {group_sizes_batch}"
                )
        elif isinstance(pred_value, Mapping):
            batch_tensor = _get_zoom_tensor(pred_value, zoom)
            if var_axis is None:
                var_axis = 1
        elif torch.is_tensor(pred_value):
            batch_tensor = pred_value
            if var_axis is None:
                var_axis = 1
        else:
            raise TypeError(f"Unsupported `{pred_key}` prediction type `{type(pred_value)}` in batch {batch_idx}.")

        batch_tensors.append(batch_tensor)

    if not batch_tensors:
        raise ValueError(f"No predictions found for key `{pred_key}`.")

    output = torch.cat(batch_tensors, dim=0)
    if group_sizes is None:
        var_dim = var_axis if var_axis is not None else (1 if output.dim() == 5 else 2)
        group_sizes = [int(output.shape[var_dim])]
    if var_axis is None:
        var_axis = 1
    return output, group_sizes, var_axis


@hydra.main(version_base=None, config_path="../configs/", config_name="mg_transformer_test")
def test(cfg: DictConfig) -> None:
    """
    Main training function that initializes datasets, dataloaders, model, and trainer,
    then begins the training process.

    :param cfg: Configuration object containing all settings for training, datasets,
                model, and logging.
    """
    if not os.path.exists(os.path.dirname(cfg.output_path)):
        os.makedirs(os.path.dirname(cfg.output_path))

    test_dataset: BaseDataset = instantiate(cfg.dataloader.dataset, data_dict=cfg.data_split["test"])

    model: Any = instantiate(cfg.model)
    trainer: Trainer = instantiate(cfg.trainer)

    if cfg.ckpt_path is not None:
        model = load_from_state_dict(model, cfg.ckpt_path, print_keys=True, device=model.device)[0]

    data_module: DataModule = instantiate(cfg.dataloader.datamodule, dataset_test=test_dataset)

    start_time = time.time()
    predictions = trainer.predict(model=model, dataloaders=data_module.test_dataloader())
    end_time = time.time()
    print(f"Predicted time: {end_time - start_time:.2f} seconds")

    if not predictions:
        raise ValueError("`trainer.predict` returned no predictions.")

    max_zoom = max(test_dataset.zooms)
    sampling = test_dataset.sampling_zooms_collate or test_dataset.sampling_zooms
    sampling = sampling[max_zoom]["zoom_patch_sample"]
    if sampling == -1:
        n_patches = 1
    else:
        npix = hp.nside2npix(2 ** max_zoom)
        n_patches = npix // 4 ** (max_zoom - sampling)

    grouped_variables, variables_flat = _group_and_variable_meta(test_dataset.data_dict)
    output_keys = list(variables_flat)
    expected_group_sizes = [len(v) for v in grouped_variables]

    output, output_group_sizes, output_var_axis = _concat_prediction_groups(
        predictions,
        "output",
        max_zoom,
        expected_group_sizes=expected_group_sizes,
    )
    mask, mask_group_sizes, mask_var_axis = _concat_prediction_groups(
        predictions,
        "mask",
        max_zoom,
        expected_group_sizes=expected_group_sizes,
    )

    if output_group_sizes != mask_group_sizes:
        raise ValueError(f"Output/mask group sizes differ: {output_group_sizes} vs {mask_group_sizes}")

    if expected_group_sizes != output_group_sizes:
        raise ValueError(
            "Prediction group sizes do not match dataset variable groups. "
            f"pred={output_group_sizes}, expected={expected_group_sizes}"
        )

    output = _merge_patch_dimension(output, n_patches, output_var_axis)
    mask = _merge_patch_dimension(mask, n_patches, mask_var_axis)

    if output.shape[1] != len(output_keys):
        raise ValueError(
            "Output variable axis does not match variable list length. "
            f"output.shape={tuple(output.shape)}, expected_vars={len(output_keys)}"
        )
    if mask.shape[1] != len(output_keys):
        raise ValueError(
            "Mask variable axis does not match variable list length. "
            f"mask.shape={tuple(mask.shape)}, expected_vars={len(output_keys)}"
        )

    normalizer_zoom = max(test_dataset.var_normalizers.keys())
    if predictions[0].get("output_var") is not None:
        output_var, output_var_group_sizes, output_var_axis = _concat_prediction_groups(
            predictions,
            "output_var",
            max_zoom,
            expected_group_sizes=expected_group_sizes,
        )
        output_var = _merge_patch_dimension(output_var, n_patches, output_var_axis)
        if output_var_group_sizes != expected_group_sizes:
            raise ValueError(
                "Output variance group sizes do not match dataset variable groups. "
                f"pred={output_var_group_sizes}, expected={expected_group_sizes}"
            )
        if output_var.shape[1] != len(output_keys):
            raise ValueError(
                "Output variance variable axis does not match variable list length. "
                f"output_var.shape={tuple(output_var.shape)}, expected_vars={len(output_keys)}"
            )
        for var_idx, var_name in enumerate(variables_flat[: output_var.shape[1]]):
            output_var[:, var_idx] = test_dataset.var_normalizers[normalizer_zoom][var_name].denormalize_var(
                output_var[:, var_idx],
                data=output[:, var_idx],
            )
        output_var_dict = dict(zip(output_keys, output_var.split(1, dim=1)))
        torch.save(output_var_dict, cfg.output_path.replace(".pt", "_var.pt"))

    for var_idx, var_name in enumerate(variables_flat[: output.shape[1]]):
        output[:, var_idx] = test_dataset.var_normalizers[normalizer_zoom][var_name].denormalize(output[:, var_idx])

    output_dict = dict(zip(output_keys, output.split(1, dim=1)))
    torch.save(output_dict, cfg.output_path)
    mask_dict = dict(zip(output_keys, mask.split(1, dim=1)))
    torch.save(mask_dict, cfg.output_path.replace(".pt", "_mask.pt"))


if __name__ == "__main__":
    test()
