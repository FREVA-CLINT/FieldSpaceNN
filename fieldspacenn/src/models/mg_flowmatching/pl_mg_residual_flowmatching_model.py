import operator
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from .pl_mg_flowmatching_model import LightningMGFlowMatchingModel
from ...utils.helpers import load_from_state_dict, merge_sampling_dicts


class LightningMGResidualFlowMatchingModel(LightningMGFlowMatchingModel):
    """Flow matching on the error of a frozen deterministic model."""

    def __init__(
        self,
        deterministic_model: nn.Module,
        deterministic_checkpoint_path: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not deterministic_checkpoint_path:
            raise ValueError("`deterministic_checkpoint_path` must be provided.")

        self.deterministic_model = deterministic_model
        self.deterministic_model, matching_keys = load_from_state_dict(
            self.deterministic_model,
            deterministic_checkpoint_path,
            device="cpu",
            print_keys=False,
        )
        if not matching_keys:
            raise ValueError(
                "The deterministic checkpoint has no parameters matching the configured model."
            )
        self._freeze_deterministic_model()

    def _freeze_deterministic_model(self) -> None:
        self.deterministic_model.requires_grad_(False)
        self.deterministic_model.eval()

    def train(self, mode: bool = True) -> "LightningMGResidualFlowMatchingModel":
        super().train(mode)
        if hasattr(self, "deterministic_model"):
            self.deterministic_model.eval()
        return self

    def _predict_deterministic(
        self,
        source_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        mask_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        emb_groups: Sequence[Optional[Dict[str, Any]]],
        sample_configs: Mapping[int, Dict[str, Any]],
    ) -> List[Optional[Dict[int, torch.Tensor]]]:
        with torch.no_grad():
            prediction_groups = self.deterministic_model(
                x_zooms_groups=self._copy_groups(source_groups),
                mask_zooms_groups=mask_groups,
                emb_groups=emb_groups,
                sample_configs=sample_configs,
            )
        return list(prediction_groups)

    @staticmethod
    def _apply_group_operation(
        left_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        right_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        operation: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> List[Optional[Dict[int, torch.Tensor]]]:
        if len(left_groups) != len(right_groups):
            raise ValueError("Prediction and target group counts must match.")

        output_groups: List[Optional[Dict[int, torch.Tensor]]] = []
        for left_group, right_group in zip(left_groups, right_groups):
            if left_group is None and right_group is None:
                output_groups.append(None)
                continue
            if not left_group or not right_group:
                raise ValueError("Prediction and target groups must have matching values.")
            if left_group.keys() != right_group.keys():
                raise ValueError("Prediction and target groups must have matching zooms.")
            output_groups.append({
                int(zoom): operation(left_group[zoom], right_group[zoom])
                for zoom in left_group
            })
        return output_groups

    def _prepare_flow_target_groups(
        self,
        source_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        target_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        mask_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        emb_groups: Sequence[Optional[Dict[str, Any]]],
        sample_configs: Mapping[int, Dict[str, Any]],
    ) -> List[Optional[Dict[int, torch.Tensor]]]:
        deterministic_groups = self._predict_deterministic(
            source_groups,
            mask_groups,
            emb_groups,
            sample_configs,
        )
        residual_groups = self._apply_group_operation(
            target_groups,
            deterministic_groups,
            operator.sub,
        )
        return self._preserve_known_source_values(
            residual_groups,
            source_groups,
            mask_groups,
        )

    @staticmethod
    def _preserve_known_source_values(
        residual_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        source_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        mask_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
    ) -> List[Optional[Dict[int, torch.Tensor]]]:
        """Keep observed source values as conditioning inside the residual flow."""
        conditioned_groups: List[Optional[Dict[int, torch.Tensor]]] = []
        for residual_group, source_group, mask_group in zip(
            residual_groups,
            source_groups,
            mask_groups,
        ):
            if not residual_group or not source_group or not mask_group:
                conditioned_groups.append(residual_group)
                continue

            conditioned_group: Dict[int, torch.Tensor] = {}
            for zoom, residual in residual_group.items():
                mask = mask_group[zoom]
                known = ~mask if mask.dtype == torch.bool else mask <= 0
                conditioned_group[int(zoom)] = torch.where(
                    known.expand_as(residual),
                    source_group[zoom],
                    residual,
                )
            conditioned_groups.append(conditioned_group)
        return conditioned_groups

    def _reconstruct_flow_prediction(
        self,
        flow_prediction_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        source_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        mask_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        emb_groups: Sequence[Optional[Dict[str, Any]]],
        sample_configs: Mapping[int, Dict[str, Any]],
    ) -> List[Optional[Dict[int, torch.Tensor]]]:
        deterministic_groups = self._predict_deterministic(
            source_groups,
            mask_groups,
            emb_groups,
            sample_configs,
        )
        prediction_groups = self._apply_group_operation(
            deterministic_groups,
            flow_prediction_groups,
            operator.add,
        )
        return self._restore_unmasked_source_values(
            prediction_groups,
            source_groups,
            mask_groups,
        )

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

        deterministic_groups = self._predict_deterministic(
            source_groups,
            mask_groups,
            emb_groups,
            sample_configs,
        )
        residual_template = [
            {int(zoom): torch.zeros_like(tensor) for zoom, tensor in group.items()}
            if group else None
            for group in deterministic_groups
        ]
        residual_template = self._preserve_known_source_values(
            residual_template,
            source_groups,
            mask_groups,
        )
        residual_groups = self._sample_flow_groups(
            residual_template,
            mask_groups,
            emb_groups,
            sample_configs,
        )
        if self.flow_matching.temporal_difference:
            residual_groups = self.flow_matching.decode_temporal_differences(
                residual_groups,
                mask_groups,
                emb_groups,
            )

        prediction_groups = self._apply_group_operation(
            deterministic_groups,
            residual_groups,
            operator.add,
        )
        prediction_groups = self._restore_unmasked_source_values(
            prediction_groups,
            source_groups,
            mask_groups,
        )
        return self._decode_prediction_groups(
            prediction_groups,
            target_groups,
            sample_configs,
        )
