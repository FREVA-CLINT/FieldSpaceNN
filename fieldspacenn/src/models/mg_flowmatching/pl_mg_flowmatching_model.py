import copy
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from lightning.pytorch.utilities import rank_zero_only

from .mg_flowmatching_model import MGFlowMatchingModel
from ..mg_transformer.pl_mg_model import LightningMGModel
from ..mg_transformer.pl_mg_probabilistic import LightningProbabilisticModel
from ...modules.flowmatching.mg_flow_matching import MGFlowMatching
from ...modules.flowmatching.mg_sampler import EulerFlowSampler
from ...modules.grids.grid_utils import decode_zooms
from ...utils.helpers import merge_sampling_dicts


class LightningMGFlowMatchingModel(LightningMGModel, LightningProbabilisticModel):
    """
    Lightning wrapper for sequential multi-block flow-matching training and inference.
    """

    def __init__(
        self,
        model: MGFlowMatchingModel,
        flow_matching: MGFlowMatching,
        lr_groups: Mapping[str, Mapping[str, Any]],
        lambda_loss_dict: Mapping[str, Any],
        data_variables: Optional[Mapping[str, Any]] = None,
        weight_decay: float = 0.0,
        sampler: str = "euler",
        n_samples: int = 1,
        max_batchsize: int = -1,
        decode_zooms: bool = True,
        sampling_steps_per_block: int = 50,
        block_loss_weights: Optional[Sequence[float]] = None,
        restore_unmasked_source_after_prediction: bool = False,
    ) -> None:
        """
        Initialize the multi-block flow-matching Lightning wrapper.
        """
        super().__init__(
            model=model,
            lr_groups=lr_groups,
            lambda_loss_dict=lambda_loss_dict,
            data_variables=data_variables,
            weight_decay=weight_decay,
        )

        self.flow_matching: MGFlowMatching = flow_matching
        if sampler != "euler":
            raise ValueError(f"`sampler` must be 'euler', got `{sampler}`.")
        self.sampler: EulerFlowSampler = EulerFlowSampler(self.flow_matching)

        self.n_samples: int = int(n_samples)
        self.max_batchsize: int = int(max_batchsize)
        self.decode_zooms: bool = bool(decode_zooms)
        self.restore_unmasked_source_after_prediction: bool = bool(
            restore_unmasked_source_after_prediction
        )
        self.sampling_steps_per_block: int = int(sampling_steps_per_block)
        if self.sampling_steps_per_block <= 0:
            raise ValueError("`sampling_steps_per_block` must be > 0.")

        if block_loss_weights is None:
            self.block_loss_weights: List[float] = [1.0 / float(self.model.n_blocks)] * self.model.n_blocks
        else:
            if len(block_loss_weights) != self.model.n_blocks:
                raise ValueError(
                    "`block_loss_weights` length must match model.n_blocks. "
                    f"Got {len(block_loss_weights)} vs {self.model.n_blocks}."
                )
            self.block_loss_weights = [float(weight) for weight in block_loss_weights]

    def forward(
        self,
        x_zooms_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        sample_configs: Mapping[int, Any] = {},
        mask_zooms_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        emb_groups: Optional[Sequence[Dict[str, Any]]] = None,
        out_zoom: Optional[int] = None,
        return_all: bool = False,
    ):
        """
        Forward call into the sequential flow-matching model.
        """
        return self.model(
            x_zooms_groups=x_zooms_groups,
            mask_zooms_groups=mask_zooms_groups,
            emb_groups=emb_groups,
            sample_configs=sample_configs,
            out_zoom=out_zoom,
            return_all=return_all,
        )

    @staticmethod
    def _copy_groups(
        groups: Sequence[Optional[Dict[int, torch.Tensor]]],
    ) -> List[Optional[Dict[int, torch.Tensor]]]:
        return [group.copy() if group else None for group in groups]

    @staticmethod
    def _get_first_valid_group(
        groups: Sequence[Optional[Dict[int, torch.Tensor]]],
    ) -> Optional[Dict[int, torch.Tensor]]:
        return next((group for group in groups if group), None)

    def _get_batch_size_and_device(
        self,
        groups: Sequence[Optional[Dict[int, torch.Tensor]]],
    ) -> Tuple[int, torch.device]:
        first_valid_group = self._get_first_valid_group(groups)
        if not first_valid_group:
            return 0, self.device
        max_zoom = max(first_valid_group.keys())
        tensor = first_valid_group[max_zoom]
        return int(tensor.shape[0]), tensor.device

    def _get_sample_configs(self, stage: str) -> Mapping[int, Any]:
        if stage == "predict":
            dataset = self.trainer.predict_dataloaders.dataset
        elif stage == "train":
            dataset = self.trainer.train_dataloader.dataset
        else:
            dataset = self.trainer.val_dataloaders.dataset
        return dataset.sampling_zooms_collate or dataset.sampling_zooms

    def _sample_block_flow_times(
        self,
        block_idx: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        time_range = self.model.get_time_range(block_idx, inference=False)
        return self.flow_matching.sample_times(batch_size, device, time_range=time_range)

    @staticmethod
    def _extract_training_losses(
        flow_outputs: Sequence[Tuple[Any, Any, Any]],
    ) -> Tuple[List[Optional[Dict[int, torch.Tensor]]], List[Optional[Dict[int, torch.Tensor]]], List[Optional[Dict[int, torch.Tensor]]]]:
        target_groups: List[Optional[Dict[int, torch.Tensor]]] = []
        output_groups: List[Optional[Dict[int, torch.Tensor]]] = []
        pred_x1_groups: List[Optional[Dict[int, torch.Tensor]]] = []

        for group_output in flow_outputs:
            if group_output is None or len(group_output) < 2:
                target_groups.append(None)
                output_groups.append(None)
                pred_x1_groups.append(None)
                continue

            target_groups.append(group_output[0])
            output_groups.append(group_output[1])
            pred_x1_groups.append(group_output[2] if len(group_output) > 2 else None)

        return target_groups, output_groups, pred_x1_groups

    def _compute_losses_from_flow_outputs(
        self,
        source_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        output_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        target_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        sample_configs: Mapping[int, Dict[str, Any]],
        mask_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]],
        emb_groups: Optional[Sequence[Optional[Dict[str, Any]]]],
        prefix: str,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        mask_groups = list(mask_groups) if mask_groups is not None else [None] * len(source_groups)
        emb_groups = list(emb_groups) if emb_groups is not None else [None] * len(source_groups)

        valid_indices = [
            idx
            for idx, (source, output, target) in enumerate(zip(source_groups, output_groups, target_groups))
            if source and output and target
        ]

        if not valid_indices:
            return torch.tensor(0.0, device=self.device), {}

        first_valid_source = source_groups[valid_indices[0]]
        assert first_valid_source is not None
        device = next(iter(first_valid_source.values())).device
        total_loss = torch.tensor(0.0, device=device)
        loss_dict_total: Dict[str, torch.Tensor] = {}

        if len(self.lambda_loss_groups) not in {0, len(source_groups)}:
            raise ValueError(
                "`lambda_loss_groups` must be empty or match the number of groups. "
                f"Got {len(self.lambda_loss_groups)} and {len(source_groups)}."
            )
        lambda_groups = (
            list(self.lambda_loss_groups)
            if len(self.lambda_loss_groups) > 0
            else [1.0] * len(source_groups)
        )

        normalizer = self._loss_normalizer(
            [target_groups[idx] for idx in valid_indices if target_groups[idx] is not None]
        )
        group_loss_inputs = []
        for idx in valid_indices:
            source = source_groups[idx]
            output = output_groups[idx]
            target = target_groups[idx]
            mask = mask_groups[idx]
            emb = emb_groups[idx]
            assert source is not None and output is not None and target is not None

            group_loss_inputs.append(
                {
                    "source": source,
                    "output": output,
                    "target": target,
                    "mask": mask,
                    "emb": emb,
                    "group_index": idx,
                    "lambda_group": float(lambda_groups[idx]),
                    "variable_weight_map": self._build_group_variable_weight_map(idx, target, emb),
                }
            )

        group_lambda_normalizer = self._group_lambda_normalizer(group_loss_inputs)

        for group_input in group_loss_inputs:
            effective_group_lambda = (
                float(group_input["lambda_group"]) / float(group_lambda_normalizer)
            )

            loss, loss_dict = self.loss_zooms(
                group_input["output"],
                group_input["target"],
                mask=group_input["mask"],
                sample_configs=sample_configs,
                prefix=f"{prefix}/",
                emb=group_input["emb"],
                variable_weight_map=group_input["variable_weight_map"],
                group_index=group_input["group_index"],
                group_lambda=effective_group_lambda,
                normalizer=normalizer,
            )
            total_loss = total_loss + loss
            self._merge_loss_dict(loss_dict_total, loss_dict)

        if self.loss_composed.has_elements:
            max_zooms = [max(target.keys()) for target in target_groups if target]
            if max_zooms:
                max_zoom = max(max_zooms)
                for group_input in group_loss_inputs:
                    output_comp = decode_zooms(
                        group_input["output"].copy(), sample_configs=sample_configs, out_zoom=max_zoom
                    )
                    target_comp = decode_zooms(
                        group_input["target"].copy(), sample_configs=sample_configs, out_zoom=max_zoom
                    )
                    mask_comp = (
                        decode_zooms(
                            group_input["mask"], sample_configs=sample_configs, out_zoom=max_zoom
                        )
                        if group_input["mask"] is not None
                        else None
                    )

                    loss, loss_dict = self.loss_composed(
                        output_comp,
                        target_comp,
                        mask=mask_comp,
                        sample_configs=sample_configs,
                        prefix=f"{prefix}/composed_",
                        emb=group_input["emb"],
                        variable_weight_map=group_input["variable_weight_map"],
                        group_index=group_input["group_index"],
                        group_lambda=(
                            float(group_input["lambda_group"]) / float(group_lambda_normalizer)
                        ),
                        normalizer=normalizer,
                    )
                    total_loss = total_loss + loss
                    self._merge_loss_dict(loss_dict_total, loss_dict)

        return total_loss, loss_dict_total

    def _run_single_block_training_loss(
        self,
        block_idx: int,
        input_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        sample_configs: Mapping[int, Dict[str, Any]],
        mask_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        emb_groups: Sequence[Optional[Dict[str, Any]]],
        prefix: str,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, device = self._get_batch_size_and_device(input_groups)
        if batch_size == 0:
            return torch.tensor(0.0, device=self.device), {}

        flow_times = self._sample_block_flow_times(block_idx, batch_size, device)
        flow_outputs = self.flow_matching.training_losses(
            self.model.get_block(block_idx),
            input_groups,
            flow_times,
            mask_groups=mask_groups,
            emb_groups=emb_groups,
            create_pred_x1=False,
            sample_configs=sample_configs,
        )

        target_groups, output_groups, _ = self._extract_training_losses(flow_outputs)
        block_loss, block_loss_dict = self._compute_losses_from_flow_outputs(
            source_groups=input_groups,
            output_groups=output_groups,
            target_groups=target_groups,
            sample_configs=sample_configs,
            mask_groups=mask_groups,
            emb_groups=emb_groups,
            prefix=f"{prefix}/block_{block_idx}",
        )
        return block_loss, block_loss_dict

    def _aggregate_block_losses(self, block_losses: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(block_losses) == 0:
            return torch.tensor(0.0, device=self.device)

        weights = torch.tensor(
            self.block_loss_weights,
            dtype=block_losses[0].dtype,
            device=block_losses[0].device,
        )
        return sum(weight * loss for weight, loss in zip(weights, block_losses))

    def training_step(
        self,
        batch: Tuple[Any, Any, Any, Any, Dict[int, torch.Tensor]],
        batch_idx: int,
    ) -> torch.Tensor:
        sample_configs = self._get_sample_configs(stage="train")
        source_groups, target_groups, mask_groups, emb_groups, patch_index_zooms = batch
        sample_configs = merge_sampling_dicts(sample_configs, patch_index_zooms)

        mask_groups = mask_groups if mask_groups is not None else [None] * len(target_groups)
        emb_groups = emb_groups if emb_groups is not None else [None] * len(target_groups)

        block_losses: List[torch.Tensor] = []
        total_loss_dict: Dict[str, torch.Tensor] = {}

        for block_idx in range(self.model.n_blocks):
            block_loss, block_loss_dict = self._run_single_block_training_loss(
                block_idx=block_idx,
                input_groups=target_groups,
                sample_configs=sample_configs,
                mask_groups=mask_groups,
                emb_groups=emb_groups,
                prefix="train",
            )
            block_losses.append(block_loss)
            total_loss_dict.update(block_loss_dict)

        total_loss = self._aggregate_block_losses(block_losses)

        self.log_dict({"train/total_loss": total_loss.item()}, prog_bar=True)
        self.log_dict(total_loss_dict, logger=True)
        return total_loss

    def validation_step(
        self,
        batch: Tuple[Any, Any, Any, Any, Dict[int, torch.Tensor]],
        batch_idx: int,
    ) -> torch.Tensor:
        sample_configs = self._get_sample_configs(stage="val")
        source_groups, target_groups, mask_groups, emb_groups, patch_index_zooms = batch
        sample_configs = merge_sampling_dicts(sample_configs, patch_index_zooms)

        mask_groups = mask_groups if mask_groups is not None else [None] * len(target_groups)
        emb_groups = emb_groups if emb_groups is not None else [None] * len(target_groups)

        block_losses: List[torch.Tensor] = []
        total_loss_dict: Dict[str, torch.Tensor] = {}

        for block_idx in range(self.model.n_blocks):
            block_loss, block_loss_dict = self._run_single_block_training_loss(
                block_idx=block_idx,
                input_groups=target_groups,
                sample_configs=sample_configs,
                mask_groups=mask_groups,
                emb_groups=emb_groups,
                prefix="val",
            )
            block_losses.append(block_loss)
            total_loss_dict.update(block_loss_dict)

        total_loss = self._aggregate_block_losses(block_losses)
        self.log_dict({"val/total_loss": total_loss.item()}, prog_bar=True)
        self.log_dict(total_loss_dict, logger=True)

        if batch_idx == 0 and rank_zero_only.rank == 0:
            self.log_healpix_tensor_plot(
                source_groups=source_groups,
                target_groups=target_groups,
                mask_groups=mask_groups,
                emb_groups=emb_groups,
                patch_index_zooms=patch_index_zooms,
            )

        return total_loss

    @staticmethod
    def _slice_batch_item(value: Any, index: int = 0) -> Any:
        if isinstance(value, torch.Tensor):
            if value.ndim > 0:
                return value[index:index + 1]
            return value
        if isinstance(value, dict):
            sliced = {}
            for key, item in value.items():
                if key == "variable_names_sampled":
                    sliced[key] = [
                        name
                        if isinstance(name, str)
                        else tuple(name[index:index + 1])
                        for name in item
                    ]
                else:
                    sliced[key] = LightningMGFlowMatchingModel._slice_batch_item(
                        item, index=index
                    )
            return sliced
        return value

    def _select_block_for_time(self, time_value: float, inference: bool = False) -> int:
        """
        Select the block responsible for a continuous time value.

        If ranges do not fully cover ``time_value``, fall back to the nearest range center.
        """
        eps = 1e-8
        best_idx = 0
        best_distance = float("inf")

        for block_idx in range(self.model.n_blocks):
            start, end = self.model.get_time_range(block_idx, inference=inference)
            if start - eps <= time_value <= end + eps:
                return block_idx

            center = 0.5 * (start + end)
            distance = abs(time_value - center)
            if distance < best_distance:
                best_distance = distance
                best_idx = block_idx

        return best_idx

    def log_healpix_tensor_plot(
        self,
        source_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        target_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        mask_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        emb_groups: Sequence[Optional[Dict[str, Any]]],
        patch_index_zooms: Dict[int, torch.Tensor],
    ) -> None:
        if not hasattr(self.logger, "log_healpix_tensor_plot"):
            return

        group_idx = next(
            (
                idx
                for idx, (source_group, target_group) in enumerate(zip(source_groups, target_groups))
                if source_group and target_group
            ),
            None,
        )
        if group_idx is None:
            return

        source_group = source_groups[group_idx]
        target_group = target_groups[group_idx]
        mask_group = mask_groups[group_idx] if mask_groups and group_idx < len(mask_groups) else None
        emb_group = emb_groups[group_idx] if emb_groups and group_idx < len(emb_groups) else None
        if source_group is None or target_group is None:
            return

        dataset = self.trainer.val_dataloaders.dataset
        plot_combined = self._dataset_applies_diff(dataset)

        max_zooms = [max(group.keys()) for group in target_groups if group]
        max_zoom = max(max_zooms) if max_zooms else max(self.model.in_zooms)

        device = source_group[max(source_group.keys())].device
        ts = torch.tensor([0.0, 0.25, 0.5, 0.75], device=device)

        source_groups_p = [
            {zoom: tensor[0:1] for zoom, tensor in group.items()} if group else None
            for group in source_groups
        ]
        target_groups_p = [
            {zoom: tensor[0:1] for zoom, tensor in group.items()} if group else None
            for group in target_groups
        ]
        mask_groups_p = [
            {zoom: tensor[0:1] for zoom, tensor in group.items()} if group else None
            for group in mask_groups
        ]
        emb_groups_p = [
            self._slice_batch_item(group, index=0) if group else None
            for group in emb_groups
        ]

        source_p = source_groups_p[group_idx]
        target_p = target_groups_p[group_idx]
        mask_p = mask_groups_p[group_idx]
        emb_p = emb_groups_p[group_idx]
        if source_p is None or target_p is None:
            return

        patch_index_zooms_p = {zoom: patch_index_zooms[zoom][0:1] for zoom in patch_index_zooms.keys()}
        sample_configs_p = merge_sampling_dicts(
            copy.deepcopy(self._get_sample_configs(stage="val")),
            patch_index_zooms_p,
        )

        noise_groups_p = [
            self.flow_matching.generate_noise(
                group, self.flow_matching._variable_names(emb_group)
            )
            if group else None
            for group, emb_group in zip(target_groups_p, emb_groups_p)
        ]
        time_dtype = next(iter(target_p.values())).dtype

        for t in ts:
            time_value = float(t.item())
            block_idx = self._select_block_for_time(time_value, inference=False)
            time_tensor = torch.tensor([time_value], device=device, dtype=time_dtype)

            pred_x1_outputs = self.flow_matching.training_losses(
                self.model.get_block(block_idx),
                target_groups_p,
                time_tensor,
                mask_groups=mask_groups_p,
                emb_groups=emb_groups_p,
                noise_groups=noise_groups_p,
                create_pred_x1=True,
                sample_configs=sample_configs_p,
            )

            _, _, pred_x1_groups = self._extract_training_losses(pred_x1_outputs)
            pred_x1_group = (
                pred_x1_groups[group_idx]
                if group_idx < len(pred_x1_groups)
                else None
            )
            if not pred_x1_group:
                continue

            if plot_combined and self.decode_zooms:
                pred_x1_comp = decode_zooms(
                    pred_x1_group.copy(),
                    sample_configs=sample_configs_p,
                    out_zoom=max_zoom,
                )
            elif plot_combined:
                pred_x1_comp = {max_zoom: pred_x1_group[max_zoom]}
            else:
                pred_x1_comp = None

            self.logger.log_healpix_tensor_plot(
                source_p,
                pred_x1_group,
                target_p,
                mask_p,
                sample_configs_p,
                emb_p,
                max_zoom,
                self.current_epoch,
                output_comp=pred_x1_comp,
                plot_name=f"_flow_{t.item():.2f}",
                plot_combined=plot_combined,
            )

    def predict_step(
        self,
        batch: Tuple[Any, Any, Any, Any, Dict[int, torch.Tensor]],
        batch_idx: int,
    ) -> Dict[str, Any]:
        return LightningProbabilisticModel.predict_step(self, batch, batch_idx)

    def _sample_block_range(
        self,
        block_idx: int,
        input_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        mask_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        emb_groups: Sequence[Optional[Dict[str, Any]]],
        sample_configs: Mapping[int, Dict[str, Any]],
        initialize_from_noise: bool = True,
    ) -> List[Optional[Dict[int, torch.Tensor]]]:
        if initialize_from_noise:
            x_t_groups = [
                self.flow_matching.generate_noise(
                    group, self.flow_matching._variable_names(emb_group)
                )
                if group else None
                for group, emb_group in zip(input_groups, emb_groups)
            ]
        else:
            x_t_groups = self._copy_groups(input_groups)

        return self.sampler.sample_loop(
            self.model.get_block(block_idx),
            input_groups=input_groups,
            x_t_groups=x_t_groups,
            mask_groups=mask_groups,
            emb_groups=emb_groups,
            sample_configs=sample_configs,
            time_range=self.model.get_time_range(block_idx, inference=True),
            n_steps=self.sampling_steps_per_block,
        )

    @staticmethod
    def _restore_unmasked_source_values(
        output_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        source_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        mask_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
    ) -> List[Optional[Dict[int, torch.Tensor]]]:
        """Replace known (unmasked) prediction values with their source values."""
        restored_groups: List[Optional[Dict[int, torch.Tensor]]] = []
        for group_idx, output_group in enumerate(output_groups):
            if not output_group:
                restored_groups.append(output_group)
                continue

            source_group = source_groups[group_idx] if group_idx < len(source_groups) else None
            mask_group = mask_groups[group_idx] if group_idx < len(mask_groups) else None
            restored_group: Dict[int, torch.Tensor] = {}
            for zoom, output in output_group.items():
                if not source_group or zoom not in source_group or not mask_group or zoom not in mask_group:
                    restored_group[int(zoom)] = output
                    continue

                mask = mask_group[zoom]
                known = ~mask if mask.dtype == torch.bool else mask <= 0
                restored_group[int(zoom)] = torch.where(
                    known.expand_as(output),
                    source_group[zoom],
                    output,
                )
            restored_groups.append(restored_group)
        return restored_groups

    def _predict_step(
        self,
        source_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        target_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        patch_index_zooms: Dict[int, torch.Tensor],
        mask_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        emb_groups: Sequence[Optional[Dict[str, Any]]],
    ) -> List[Optional[Dict[int, torch.Tensor]]]:
        sample_configs = self._get_sample_configs(stage="predict")
        sample_configs = merge_sampling_dicts(sample_configs, patch_index_zooms)

        mask_groups = mask_groups if mask_groups is not None else [None] * len(source_groups)
        emb_groups = emb_groups if emb_groups is not None else [None] * len(source_groups)

        current_groups = self._copy_groups(source_groups)

        for block_idx in range(self.model.n_blocks):
            current_groups = self._sample_block_range(
                block_idx=block_idx,
                input_groups=current_groups,
                mask_groups=mask_groups,
                emb_groups=emb_groups,
                sample_configs=sample_configs,
                initialize_from_noise=(block_idx == 0),
            )

        if self.restore_unmasked_source_after_prediction:
            current_groups = self._restore_unmasked_source_values(
                current_groups,
                source_groups,
                mask_groups,
            )

        if not self.decode_zooms:
            return current_groups

        max_zoom = None
        first_target_group = self._get_first_valid_group(target_groups)
        if first_target_group:
            max_zoom = max(first_target_group.keys())
        elif len(self.model.in_zooms) > 0:
            max_zoom = max(self.model.in_zooms)

        if max_zoom is None:
            return current_groups

        decoded_outputs: List[Optional[Dict[int, torch.Tensor]]] = []
        for group in current_groups:
            if group:
                decoded_outputs.append(
                    decode_zooms(group.copy(), sample_configs=sample_configs, out_zoom=max_zoom)
                )
            else:
                decoded_outputs.append(None)
        return decoded_outputs
