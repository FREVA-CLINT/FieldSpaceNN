from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, List

import torch
from pytorch_lightning.utilities import rank_zero_only

from ...utils.helpers import merge_sampling_dicts
from ...modules.grids.grid_utils import decode_zooms
from .pl_mg_model import LightningMGModel


class LightningMGAutoregressiveModel(LightningMGModel):
    def __init__(
        self,
        model: Any,
        lr_groups: Mapping[str, Mapping[str, Any]],
        lambda_loss_dict: Mapping[str, Any],
        weight_decay: float = 0.0,
        lambda_loss_groups: list = [],
        lambda_loss_autoregressive: Optional[Sequence[float]] = None,
        return_all_steps: bool = False,
    ) -> None:
        super().__init__(
            model=model,
            lr_groups=lr_groups,
            lambda_loss_dict=lambda_loss_dict,
            weight_decay=weight_decay,
            lambda_loss_groups=lambda_loss_groups,
        )
        if lambda_loss_autoregressive is None:
            lambda_loss_autoregressive = [1.0]
        self.lambda_loss_autoregressive = [float(x) for x in lambda_loss_autoregressive]
        self.return_all_steps = return_all_steps

    @property
    def n_autoregressive_steps(self) -> int:
        return len(self.lambda_loss_autoregressive)

    @staticmethod
    def _extract_forecast_groups(
        groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        n_steps: int,
    ) -> Sequence[Optional[Dict[int, torch.Tensor]]]:
        forecast_groups = []
        for group in groups:
            if group is None:
                forecast_groups.append(None)
                continue

            forecast_group = {}
            for zoom, tensor in group.items():
                if tensor.shape[2] < n_steps:
                    raise ValueError(
                        f"Cannot extract {n_steps} forecast steps for zoom {zoom}: "
                        f"tensor only has length {tensor.shape[2]}."
                    )
                forecast_group[zoom] = tensor[:, :, -n_steps:]
            forecast_groups.append(forecast_group)

        return forecast_groups

    @staticmethod
    def _append_last_timestep_groups(
        history_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        output_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
    ) -> Sequence[Optional[Dict[int, torch.Tensor]]]:
        composed_groups = []
        for group_idx, history_group in enumerate(history_groups):
            if history_group is None:
                composed_groups.append(None)
                continue

            output_group = output_groups[group_idx] if group_idx < len(output_groups) and output_groups[group_idx] is not None else {}
            composed_group = {}
            for zoom, history in history_group.items():
                output = output_group.get(zoom)
                if output is None:
                    composed_group[zoom] = history
                    continue

                composed_group[zoom] = torch.concat((history, output[:, :, [-1]]), dim=2)
            composed_groups.append(composed_group)

        return composed_groups

    def forward_autoregressive(
        self,
        x_zooms_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        mask_zooms_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        emb_groups: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
        sample_configs: Mapping[int, Dict[str, Any]] = {},
        out_zoom: Optional[int] = None,
        mask_zooms: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        emb: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
        n_steps: int = 0,
        return_mode: str = "forecast",
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

        if return_mode not in {"all", "last", "forecast"}:
            raise ValueError(
                f"Unsupported autoregressive return_mode '{return_mode}'. "
                "Expected one of {'all', 'last', 'forecast'}."
            )

        if n_steps <= 0:
            if return_mode == "all":
                return []
            return list(x_zooms_groups)

        source_groups = []
        current_groups = []
        for group in x_zooms_groups:
            if group is None:
                source_groups.append(None)
                current_groups.append(None)
            else:
                cloned_group = {int(zoom): tensor.clone() for zoom, tensor in group.items()}
                source_groups.append({zoom: tensor.clone() for zoom, tensor in cloned_group.items()})
                current_groups.append(cloned_group)

        current_masks = None
        if mask_zooms_groups is not None:
            current_masks = []
            for group in mask_zooms_groups:
                if group is None:
                    current_masks.append(None)
                else:
                    current_masks.append({int(zoom): tensor.clone() for zoom, tensor in group.items()})

        current_emb_groups = None
        if emb_groups is not None:
            current_emb_groups = []
            for group in emb_groups:
                if group is None:
                    current_emb_groups.append(None)
                    continue

                emb_group = {}
                for key, value in group.items():
                    if isinstance(value, dict):
                        emb_group[key] = {
                            int(zoom) if isinstance(zoom, (int, str)) and str(zoom).lstrip("-").isdigit() else zoom:
                            tensor.clone() if torch.is_tensor(tensor) else tensor
                            for zoom, tensor in value.items()
                        }
                    else:
                        emb_group[key] = value.clone() if torch.is_tensor(value) else value
                current_emb_groups.append(emb_group)

        dataset_train = getattr(getattr(getattr(self, "trainer", None), "datamodule", None), "dataset_train", None)
        mask_ts_mode = getattr(dataset_train, "mask_ts_mode", "repeat")

        output_steps = []
        composed_groups = source_groups
        for _ in range(n_steps):
            output_groups = self(
                x_zooms_groups=current_groups,
                mask_zooms_groups=current_masks,
                emb_groups=current_emb_groups,
                sample_configs=sample_configs,
                out_zoom=out_zoom,
                **kwargs,
            )
            composed_groups = self._append_last_timestep_groups(composed_groups, output_groups)
            output_steps.append(composed_groups)

            next_groups = []
            for group_idx, current_group in enumerate(current_groups):
                if current_group is None:
                    next_groups.append(None)
                    continue

                output_zooms = output_groups[group_idx] if group_idx < len(output_groups) and output_groups[group_idx] is not None else {}
                next_group = {}
                for zoom, current in current_group.items():
                    if zoom not in output_zooms:
                        next_group[zoom] = current
                        continue

                    output = output_zooms[zoom]
                    rolled = output.clone() if output.shape[2] <= 1 else torch.concat((output[:, :, 1:], output[:, :, [-1]]), dim=2)

                    if mask_ts_mode == 'zero':
                        rolled[:, :, -1] = 0

                    next_group[zoom] = rolled

                next_groups.append(next_group)
            current_groups = next_groups

            if current_emb_groups is not None:
                shifted_emb_groups = []
                for group in current_emb_groups:
                    if group is None:
                        shifted_emb_groups.append(None)
                        continue

                    emb_group = dict(group)
                    if 'TimeEmbedder' in group and isinstance(group['TimeEmbedder'], dict):
                        emb_group['TimeEmbedder'] = shift_timeembedding(group['TimeEmbedder'])
                    shifted_emb_groups.append(emb_group)
                current_emb_groups = shifted_emb_groups
        if return_mode == "all":
            return output_steps

        last_output_groups = output_steps[-1] if output_steps else list(x_zooms_groups)
        if return_mode == "last":
            return last_output_groups

        return self._extract_forecast_groups(last_output_groups, n_steps=n_steps)

    def get_losses(
        self,
        source_groups: Sequence[Dict[int, torch.Tensor]] | Dict[int, torch.Tensor],
        target_groups: Sequence[Dict[int, torch.Tensor]],
        sample_configs: Mapping[int, Dict[str, Any]] = {},
        sample_configs_target: Optional[Mapping[int, Dict[str, Any]]] = None,
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
            n_steps=self.n_autoregressive_steps,
            return_mode="forecast",
        )
        forecast_mask_groups = None
        if mask_groups is not None:
            forecast_mask_groups = self._extract_forecast_groups(
                mask_groups,
                n_steps=self.n_autoregressive_steps,
            )

        return self._compute_losses_from_output_groups(
            source_groups=source_groups_list,
            output_groups=output_groups,
            target_groups=target_groups,
            sample_configs=sample_configs,
            sample_configs_target=sample_configs_target,
            mask_groups=forecast_mask_groups,
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
        source_groups, target_groups, mask_groups, emb_groups, patch_index_zooms = batch

        sample_configs = merge_sampling_dicts(sample_configs, patch_index_zooms)
        sample_configs_target = merge_sampling_dicts(sample_configs_target, patch_index_zooms)

        loss, loss_dict, _ = self.get_losses(
            source_groups,
            target_groups,
            sample_configs,
            sample_configs_target=sample_configs_target,
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
        source_groups, target_groups, mask_groups, emb_groups, patch_index_zooms = batch

        max_zooms = [max(target.keys()) for target in target_groups if target]
        max_zoom = max(max_zooms) if max_zooms else max(self.model.in_zooms)

        sample_configs = merge_sampling_dicts(sample_configs, patch_index_zooms)
        sample_configs_target = merge_sampling_dicts(sample_configs_target, patch_index_zooms)

        loss, loss_dict, output_groups = self.get_losses(
            [group.copy() for group in source_groups],
            target_groups,
            sample_configs=sample_configs,
            sample_configs_target=sample_configs_target,
            mask_groups=mask_groups,
            emb_groups=emb_groups,
            prefix='val',
        )

        self.log_dict({"validate/total_loss": loss.item()}, prog_bar=True)
        self.log_dict(loss_dict, logger=True)

        if batch_idx == 0 and rank_zero_only.rank == 0:
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
        sample_configs = self.trainer.predict_dataloaders.dataset.sampling_zooms_collate or self.trainer.predict_dataloaders.dataset.sampling_zooms
        source_groups, target_groups, mask_groups, emb_groups, patch_index_zooms = batch

        sample_configs = merge_sampling_dicts(sample_configs, patch_index_zooms)
        output = self.forward_autoregressive(
            x_zooms_groups=[group.copy() if group is not None else None for group in source_groups],
            mask_zooms_groups=mask_groups,
            emb_groups=emb_groups,
            sample_configs=sample_configs,
            n_steps=self.n_autoregressive_steps,
            return_mode="all" if self.return_all_steps else "forecast",
        )

        return {
            "output": output,
            "mask": mask_groups,
            "target": target_groups,
        }


def shift_timeembedding(emb_time: Dict[int, torch.Tensor]):
    emb_time = emb_time.copy()
    for zoom, time_zoom in emb_time.items():
        emb_time[zoom] = time_zoom + time_zoom.diff().mean()

    return emb_time
