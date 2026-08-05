from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, List

import torch
from pytorch_lightning.utilities import rank_zero_only

from ...utils.helpers import merge_sampling_dicts
from ...modules.grids.grid_utils import decode_zooms
from .pl_mg_model import LightningMGModel

ROLLOUT_TIME_ALIGNED_EMBED_KEYS = (
    "StaticVariableEmbedder",
    "TimeEmbedder",
    "TimeProgressEmbedder",
    "TimeIndexEmbedder",
)


def _clone_embedding_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, dict):
        cloned_dict = {}
        for key, item in value.items():
            normalized_key = (
                int(key) if isinstance(key, (int, str)) and str(key).lstrip("-").isdigit() else key
            )
            cloned_dict[normalized_key] = _clone_embedding_value(item)
        return cloned_dict
    return value


def clone_embedding_groups(
    emb_groups: Optional[Sequence[Optional[Dict[str, Any]]]],
) -> Optional[List[Optional[Dict[str, Any]]]]:
    if emb_groups is None:
        return None

    cloned_groups: List[Optional[Dict[str, Any]]] = []
    for group in emb_groups:
        if group is None:
            cloned_groups.append(None)
            continue

        cloned_groups.append({
            key: _clone_embedding_value(value)
            for key, value in group.items()
        })

    return cloned_groups


def get_rollout_embedding_time_slice(
    zoom: int,
    sample_configs_emb: Mapping[int, Dict[str, Any]],
    sample_configs_source: Mapping[int, Dict[str, Any]],
    step_idx: int,
    emb_time_len: int,
) -> slice:
    if zoom not in sample_configs_source:
        raise ValueError(f"Missing source sampling config for zoom {zoom}.")
    if zoom not in sample_configs_emb:
        raise ValueError(f"Missing embedding sampling config for zoom {zoom}.")

    source_cfg = sample_configs_source[zoom]
    emb_cfg = sample_configs_emb[zoom]
    source_n_past = int(source_cfg["n_past_ts"])
    source_n_future = int(source_cfg["n_future_ts"])
    emb_n_past = int(emb_cfg["n_past_ts"])
    emb_n_future = int(emb_cfg["n_future_ts"])

    source_time_len = source_n_past + source_n_future + 1
    start = emb_n_past - source_n_past + int(step_idx)
    end = start + source_time_len

    if start < 0 or end > emb_time_len:
        raise ValueError(
            f"Autoregressive rollout step {step_idx} at zoom {zoom} requires embedding time range "
            f"[{start}, {end}) inside a loaded embedding window of length {emb_time_len}. "
            f"Source window uses n_past_ts={source_n_past}, n_future_ts={source_n_future}; "
            f"embedding window uses n_past_ts={emb_n_past}, n_future_ts={emb_n_future}."
        )

    return slice(start, end)


def slice_rollout_embedding_zoom_map(
    embedding_map: Mapping[int, torch.Tensor],
    sample_configs_emb: Mapping[int, Dict[str, Any]],
    sample_configs_source: Mapping[int, Dict[str, Any]],
    step_idx: int,
) -> Dict[int, torch.Tensor]:
    aligned_embeddings: Dict[int, torch.Tensor] = {}
    for zoom_key, tensor in embedding_map.items():
        zoom = int(zoom_key)
        if tensor.ndim < 2:
            aligned_embeddings[zoom] = tensor.clone()
            continue

        time_slice = get_rollout_embedding_time_slice(
            zoom=zoom,
            sample_configs_emb=sample_configs_emb,
            sample_configs_source=sample_configs_source,
            step_idx=step_idx,
            emb_time_len=int(tensor.shape[1]),
        )
        aligned_embeddings[zoom] = tensor[:, time_slice].clone()

    return aligned_embeddings


def align_embedding_groups_to_rollout_step(
    base_emb_groups: Optional[Sequence[Optional[Dict[str, Any]]]],
    sample_configs_emb: Mapping[int, Dict[str, Any]],
    sample_configs_source: Mapping[int, Dict[str, Any]],
    step_idx: int,
) -> Optional[List[Optional[Dict[str, Any]]]]:
    if base_emb_groups is None:
        return None

    aligned_groups: List[Optional[Dict[str, Any]]] = []
    for group in base_emb_groups:
        if group is None:
            aligned_groups.append(None)
            continue

        aligned_group: Dict[str, Any] = {}
        for key, value in group.items():
            if key in ROLLOUT_TIME_ALIGNED_EMBED_KEYS and isinstance(value, Mapping):
                aligned_group[key] = slice_rollout_embedding_zoom_map(
                    embedding_map=value,
                    sample_configs_emb=sample_configs_emb,
                    sample_configs_source=sample_configs_source,
                    step_idx=step_idx,
                )
            else:
                aligned_group[key] = _clone_embedding_value(value)
        aligned_groups.append(aligned_group)

    return aligned_groups


def _unwrap_dataset(dataset: Optional[Any]) -> Optional[Any]:
    current = dataset
    while current is not None and hasattr(current, "dataset"):
        nested = getattr(current, "dataset")
        if nested is None or nested is current:
            break
        current = nested
    return current


class LightningMGAutoregressiveModel(LightningMGModel):
    def __init__(
        self,
        model: Any,
        lr_groups: Mapping[str, Mapping[str, Any]],
        lambda_loss_dict: Mapping[str, Any],
        data_variables: Optional[Mapping[str, Any]] = None,
        weight_decay: float = 0.0,
        lambda_loss_groups: list = [],
        n_autoregressive_steps: int = 1,
        return_all_steps: bool = False,
    ) -> None:
        super().__init__(
            model=model,
            lr_groups=lr_groups,
            lambda_loss_dict=lambda_loss_dict,
            data_variables=data_variables,
            weight_decay=weight_decay,
            lambda_loss_groups=lambda_loss_groups,
        )
        if n_autoregressive_steps < 1:
            raise ValueError("n_autoregressive_steps must be at least 1.")
        self.n_autoregressive_steps = int(n_autoregressive_steps)
        self.return_all_steps = return_all_steps

    def _get_active_dataset(self) -> Optional[Any]:
        trainer = getattr(self, "trainer", None)
        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is None:
            predict_dataloaders = getattr(trainer, "predict_dataloaders", None)
            predict_dataset = _unwrap_dataset(getattr(predict_dataloaders, "dataset", None))
            return predict_dataset

        for dataset_name in ("dataset_train", "dataset_val", "dataset_predict", "dataset_test"):
            dataset = _unwrap_dataset(getattr(datamodule, dataset_name, None))
            if dataset is not None:
                return dataset

        return None

    def forward_autoregressive(
        self,
        x_zooms_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        mask_zooms_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        emb_groups: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
        sample_configs: Mapping[int, Dict[str, Any]] = {},
        sample_configs_emb: Optional[Mapping[int, Dict[str, Any]]] = None,
        out_zoom: Optional[int] = None,
        mask_zooms: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        emb: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
        n_steps: int = 0,
        return_all_steps: bool = False,
        **kwargs: Any,
    ) -> Sequence[Optional[Dict[int, torch.Tensor]]] | List[Sequence[Optional[Dict[int, torch.Tensor]]]]:
        if x_zooms_groups is None:
            x_zooms_groups = []
        if isinstance(x_zooms_groups, dict):
            x_zooms_groups = [x_zooms_groups]

        if mask_zooms_groups is None:
            mask_zooms_groups = mask_zooms
        if emb_groups is None:
            emb_groups = emb
        if sample_configs_emb is None:
            sample_configs_emb = sample_configs

        if n_steps <= 0:
            if return_all_steps:
                return []
            return list(x_zooms_groups)

        current_groups = []
        for group in x_zooms_groups:
            if group is None:
                current_groups.append(None)
            else:
                current_groups.append({int(zoom): tensor.clone() for zoom, tensor in group.items()})

        current_masks = None
        if mask_zooms_groups is not None:
            current_masks = []
            for group in mask_zooms_groups:
                if group is None:
                    current_masks.append(None)
                else:
                    current_masks.append({int(zoom): tensor.clone() for zoom, tensor in group.items()})

        base_emb_groups = clone_embedding_groups(emb_groups)

        dataset = self._get_active_dataset()
        mask_ts_mode = getattr(dataset, "mask_ts_mode", "repeat")
        target_time_shift = getattr(dataset, "target_time_shift", 0)

        output_steps = [] if return_all_steps else None
        forecast_groups = None
        if not return_all_steps:
            forecast_groups = []
            for group in current_groups:
                if group is None:
                    forecast_groups.append(None)
                else:
                    forecast_groups.append({zoom: [] for zoom in group})

        for step_idx in range(n_steps):
            current_emb_groups = align_embedding_groups_to_rollout_step(
                base_emb_groups=base_emb_groups,
                sample_configs_emb=sample_configs_emb,
                sample_configs_source=sample_configs,
                step_idx=step_idx,
            )
            model_output_groups = self(
                x_zooms_groups=current_groups,
                mask_zooms_groups=current_masks,
                emb_groups=current_emb_groups,
                sample_configs=sample_configs,
                out_zoom=out_zoom,
                **kwargs,
            )

            next_groups = []
            for group_idx, current_group in enumerate(current_groups):
                if current_group is None:
                    next_groups.append(None)
                    continue

                output_zooms = (
                    model_output_groups[group_idx]
                    if group_idx < len(model_output_groups) and model_output_groups[group_idx] is not None
                    else {}
                )
                next_group = {}
                forecast_group = (
                    forecast_groups[group_idx]
                    if forecast_groups is not None
                    else None
                )
                for zoom, current in current_group.items():
                    if zoom not in output_zooms:
                        next_group[zoom] = current
                        continue

                    output = output_zooms[zoom]
                    last_output = output[:, :, [-1]]

                    if forecast_group is not None:
                        forecast_group[zoom].append(last_output)

                    if target_time_shift == 0:
                        rolled = torch.concat((output[:, :, 1:], last_output), dim=2)

                        if mask_ts_mode == 'zero':
                            rolled[:, :, -1] = 0

                    else:
                        rolled = torch.concat((current[:, :, 1:], last_output), dim=2)

                    next_group[zoom] = rolled

                next_groups.append(next_group)
            if output_steps is not None:
                output_steps.append(next_groups)
            current_groups = next_groups
        if output_steps is not None:
            return output_steps

        concatenated_forecast_groups = []
        for group in forecast_groups:
            if group is None:
                concatenated_forecast_groups.append(None)
                continue
            concatenated_forecast_groups.append({
                zoom: torch.concat(outputs, dim=2)
                for zoom, outputs in group.items()
                if len(outputs) > 0
            })

        return concatenated_forecast_groups

    def get_losses(
        self,
        source_groups: Sequence[Dict[int, torch.Tensor]] | Dict[int, torch.Tensor],
        target_groups: Sequence[Dict[int, torch.Tensor]],
        sample_configs: Mapping[int, Dict[str, Any]] = {},
        sample_configs_target: Optional[Mapping[int, Dict[str, Any]]] = None,
        sample_configs_emb: Optional[Mapping[int, Dict[str, Any]]] = None,
        mask_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        emb_groups: Optional[Sequence[Dict[str, Any]]] = None,
        prefix: str = '',
        mask_zooms: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        emb: Optional[Sequence[Dict[str, Any]]] = None,
    ):
        if mask_groups is None:
            mask_groups = mask_zooms
        if emb_groups is None:
            emb_groups = emb
        if sample_configs_target is None:
            sample_configs_target = sample_configs

        if isinstance(source_groups, dict):
            source_groups_list = [source_groups]
        else:
            source_groups_list = list(source_groups)

        output_groups = self.forward_autoregressive(
            x_zooms_groups=[group.copy() if group is not None else None for group in source_groups_list],
            mask_zooms_groups=mask_groups,
            emb_groups=emb_groups,
            sample_configs=sample_configs,
            sample_configs_emb=sample_configs_emb,
            n_steps=self.n_autoregressive_steps,
        )

        return self._compute_losses_from_output_groups(
            source_groups=source_groups_list,
            output_groups=output_groups,
            target_groups=target_groups,
            sample_configs=sample_configs,
            sample_configs_target=sample_configs_target,
            mask_groups=mask_groups,
            emb_groups=emb_groups,
            prefix=prefix,
        )

    def training_step(
        self,
        batch: Tuple[Any, Any, Any, Any, Dict[int, torch.Tensor]],
        batch_idx: int,
    ) -> torch.Tensor:
        dataset = self.trainer.datamodule.dataset_train
        sample_configs = dataset.sampling_zooms_collate or dataset.sampling_zooms
        sample_configs_target = getattr(dataset, "sampling_zooms_target", sample_configs)
        sample_configs_emb = getattr(dataset, "sample_configs_emb", sample_configs)
        source_groups, target_groups, mask_groups, emb_groups, patch_index_zooms = batch

        sample_configs = merge_sampling_dicts(sample_configs, patch_index_zooms)
        sample_configs_target = merge_sampling_dicts(sample_configs_target, patch_index_zooms)
        sample_configs_emb = merge_sampling_dicts(sample_configs_emb, patch_index_zooms)

        loss, loss_dict, _ = self.get_losses(
            source_groups,
            target_groups,
            sample_configs,
            sample_configs_target=sample_configs_target,
            sample_configs_emb=sample_configs_emb,
            mask_groups=mask_groups,
            emb_groups=emb_groups,
            prefix='train',
        )

        self.log_dict({"train/total_loss": loss.item()}, prog_bar=True)
        self.log_dict(loss_dict, logger=True)
        return loss

    def validation_step(
        self,
        batch: Tuple[Any, Any, Any, Any, Dict[int, torch.Tensor]],
        batch_idx: int,
    ) -> torch.Tensor:
        dataset = self.trainer.datamodule.dataset_val
        sample_configs = dataset.sampling_zooms_collate or dataset.sampling_zooms
        sample_configs_target = getattr(dataset, "sampling_zooms_target", sample_configs)
        sample_configs_emb = getattr(dataset, "sample_configs_emb", sample_configs)
        source_groups, target_groups, mask_groups, emb_groups, patch_index_zooms = batch

        max_zooms = [max(target.keys()) for target in target_groups if target]
        max_zoom = max(max_zooms) if max_zooms else max(self.model.in_zooms)

        sample_configs = merge_sampling_dicts(sample_configs, patch_index_zooms)
        sample_configs_target = merge_sampling_dicts(sample_configs_target, patch_index_zooms)
        sample_configs_emb = merge_sampling_dicts(sample_configs_emb, patch_index_zooms)

        loss, loss_dict, output_groups = self.get_losses(
            [group.copy() for group in source_groups],
            target_groups,
            sample_configs=sample_configs,
            sample_configs_target=sample_configs_target,
            sample_configs_emb=sample_configs_emb,
            mask_groups=mask_groups,
            emb_groups=emb_groups,
            prefix='val',
        )

        self.log_dict({"validate/total_loss": loss.item()}, prog_bar=True)
        self.log_dict(loss_dict, logger=True)

        if (
            batch_idx == 0
            and rank_zero_only.rank == 0
            and self._dataset_applies_diff(dataset)
        ):
            group_idx = next((idx for idx, group in enumerate(output_groups) if group), None)
            if group_idx is None:
                return loss

            output = output_groups[group_idx]
            source = source_groups[group_idx]
            target = target_groups[group_idx]
            mask = mask_groups[group_idx]
            emb = emb_groups[group_idx]

            output_comp = decode_zooms(output.copy(), sample_configs=sample_configs_target, out_zoom=max_zoom)
            source_comp = decode_zooms(source.copy(), sample_configs=sample_configs, out_zoom=max_zoom)
            target_comp = decode_zooms(target, sample_configs=sample_configs_target, out_zoom=max_zoom)

            self.logger.log_tensor_plot(
                input=source_comp,
                output=output_comp,
                gt=target_comp,
                mask={max_zoom: mask[max_zoom]} if mask is not None and max_zoom in mask else None,
                sample_configs=sample_configs,
                emb=emb,
                plot_name=f"epoch_{self.current_epoch}_combined",
            )

        return loss

    def predict_step(
        self,
        batch: Tuple[Any, Any, Any, Any, Dict[int, torch.Tensor]],
        batch_idx: int,
    ) -> Dict[str, Any]:
        dataset = _unwrap_dataset(self.trainer.predict_dataloaders.dataset)
        sample_configs = dataset.sampling_zooms_collate or dataset.sampling_zooms
        sample_configs_target = getattr(dataset, "sampling_zooms_target", sample_configs)
        sample_configs_emb = getattr(dataset, "sample_configs_emb", sample_configs)
        source_groups, target_groups, mask_groups, emb_groups, patch_index_zooms = batch

        sample_configs = merge_sampling_dicts(sample_configs, patch_index_zooms)
        sample_configs_target = merge_sampling_dicts(sample_configs_target, patch_index_zooms)
        sample_configs_emb = merge_sampling_dicts(sample_configs_emb, patch_index_zooms)
        output = self.forward_autoregressive(
            x_zooms_groups=[group.copy() if group is not None else None for group in source_groups],
            mask_zooms_groups=mask_groups,
            emb_groups=emb_groups,
            sample_configs=sample_configs,
            sample_configs_emb=sample_configs_emb,
            n_steps=self.n_autoregressive_steps,
            return_all_steps=self.return_all_steps,
        )

        combined_zoom = max(sample_configs_target.keys())
        initial_input = []
        for group in source_groups:
            if group is None:
                initial_input.append(None)
                continue
            initial_input.append({zoom: tensor[:, :, [-1]].clone() for zoom, tensor in group.items()})

        output_combined = []
        for group in output:
            if group is None:
                output_combined.append(None)
                continue
            output_combined.append(
                decode_zooms(group.copy(), sample_configs=sample_configs_target, out_zoom=combined_zoom)
            )

        initial_input_combined = []
        for group in initial_input:
            if group is None:
                initial_input_combined.append(None)
                continue
            initial_input_combined.append(
                decode_zooms(group.copy(), sample_configs=sample_configs_target, out_zoom=combined_zoom)
            )

        return {
            "initial_input": initial_input,
            "initial_input_combined": initial_input_combined,
            "output": output,
            "output_combined": output_combined,
            "patch_index_zooms": patch_index_zooms,
        }
