import math
import operator
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from .pl_mg_flowmatching_model import LightningMGFlowMatchingModel
from ...utils.helpers import load_from_state_dict, merge_sampling_dicts
from ...utils.two_channel_prediction import split_mean_and_error_scale


class LightningMGResidualFlowMatchingModel(LightningMGFlowMatchingModel):
    """Flow matching on the error of a frozen deterministic model."""

    def __init__(
        self,
        deterministic_model: nn.Module,
        deterministic_checkpoint_path: str,
        deterministic_zooms: Sequence[int],
        spatial_conditioning_embed_key: Optional[str] = None,
        spatial_noise_scale: float = 1.0,
        scale_std_by_one_minus_tau: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not deterministic_checkpoint_path:
            raise ValueError("`deterministic_checkpoint_path` must be provided.")
        self.deterministic_zooms: set[int] = {
            int(zoom) for zoom in deterministic_zooms
        }
        if not self.deterministic_zooms:
            raise ValueError("`deterministic_zooms` must contain at least one zoom.")
        self.spatial_conditioning_embed_key = (
            None
            if spatial_conditioning_embed_key is None
            else str(spatial_conditioning_embed_key)
        )
        self.spatial_noise_scale = float(spatial_noise_scale)
        self.scale_std_by_one_minus_tau = bool(scale_std_by_one_minus_tau)
        if (
            not math.isfinite(self.spatial_noise_scale)
            or self.spatial_noise_scale < 0
        ):
            raise ValueError("`spatial_noise_scale` must be finite and nonnegative.")
        if self.flow_matching.conditioning_embed_key is None:
            self.flow_matching.conditioning_embed_key = (
                self.spatial_conditioning_embed_key
            )
        self.flow_matching.scale_std_by_one_minus_tau = (
            self.scale_std_by_one_minus_tau
        )
        configured_conditioning_keys = {
            key
            for key in (
                self.flow_matching.noise_scale_embed_key,
                self.flow_matching.conditioning_embed_key,
            )
            if key is not None
        }
        if (
            configured_conditioning_keys or self.scale_std_by_one_minus_tau
        ) and self.spatial_conditioning_embed_key is None:
            raise ValueError(
                "Spatial noise or input conditioning requires "
                "`spatial_conditioning_embed_key`."
            )
        if configured_conditioning_keys - {self.spatial_conditioning_embed_key}:
            raise ValueError(
                "Flow-matching spatial-conditioning keys must match "
                "`spatial_conditioning_embed_key`."
            )

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
    ) -> tuple[
        List[Optional[Dict[int, torch.Tensor]]],
        List[Optional[Dict[int, torch.Tensor]]],
    ]:
        with torch.no_grad():
            prediction_groups = self.deterministic_model(
                x_zooms_groups=self._copy_groups(source_groups),
                mask_zooms_groups=mask_groups,
                emb_groups=emb_groups,
                sample_configs=sample_configs,
            )
        mean_groups: List[Optional[Dict[int, torch.Tensor]]] = []
        sigma_groups: List[Optional[Dict[int, torch.Tensor]]] = []
        channels = {
            int(prediction.shape[-1])
            for prediction_group in prediction_groups
            if prediction_group
            for zoom, prediction in prediction_group.items()
            if int(zoom) in self.deterministic_zooms
        }
        if len(channels) != 1 or next(iter(channels)) not in {1, 2}:
            raise ValueError(
                "Configured deterministic zooms present in the batch must "
                "consistently output one mean channel or two mean/error channels."
            )
        has_error_scale = next(iter(channels)) == 2

        for group_index, prediction_group in enumerate(prediction_groups):
            if not prediction_group:
                mean_groups.append(prediction_group)
                sigma_groups.append(prediction_group)
                continue

            source_group = source_groups[group_index]
            if not source_group:
                raise ValueError(
                    "Deterministic output requires a matching source group."
                )
            means: Dict[int, torch.Tensor] = {}
            sigmas: Optional[Dict[int, torch.Tensor]] = (
                {} if has_error_scale else None
            )
            for zoom, source in source_group.items():
                zoom = int(zoom)
                if zoom not in self.deterministic_zooms:
                    means[zoom] = source
                    if sigmas is not None:
                        sigmas[zoom] = torch.ones_like(source)
                    continue

                if zoom not in prediction_group:
                    raise ValueError(
                        f"Deterministic model output is missing zoom {zoom}."
                    )
                prediction = prediction_group[zoom].detach()
                if has_error_scale:
                    mean, sigma = split_mean_and_error_scale(prediction)
                    # ``mean`` is a view into the two-channel prediction. Clone it
                    # so retaining the mean does not also retain log-variance storage.
                    means[zoom] = mean.clone()
                    assert sigmas is not None
                    sigmas[zoom] = sigma * self.spatial_noise_scale
                else:
                    means[zoom] = prediction
            mean_groups.append(means)
            sigma_groups.append(sigmas)
        return mean_groups, sigma_groups

    @staticmethod
    def _build_spatial_conditioning(
        mean_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        sigma_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
    ) -> List[Optional[Dict[str, Dict[int, torch.Tensor]]]]:
        conditioning_groups: List[
            Optional[Dict[str, Dict[int, torch.Tensor]]]
        ] = []
        for mean_group, sigma_group in zip(mean_groups, sigma_groups):
            if mean_group is None:
                conditioning_groups.append(None)
                continue
            conditioning = {"mu": mean_group}
            if sigma_group is not None:
                conditioning["s"] = sigma_group
            conditioning_groups.append(conditioning)
        return conditioning_groups

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

    def _prepare_residual_target_and_conditioning(
        self,
        source_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        target_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        mask_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        emb_groups: Sequence[Optional[Dict[str, Any]]],
        sample_configs: Mapping[int, Dict[str, Any]],
    ) -> tuple[
        List[Optional[Dict[int, torch.Tensor]]],
        List[Optional[Dict[str, Dict[int, torch.Tensor]]]],
    ]:
        deterministic_groups, sigma_groups = self._predict_deterministic(
            source_groups,
            mask_groups,
            emb_groups,
            sample_configs,
        )
        conditioning_groups = self._build_spatial_conditioning(
            deterministic_groups,
            sigma_groups,
        )
        residual_groups = self._apply_group_operation(
            target_groups,
            deterministic_groups,
            operator.sub,
        )
        return (
            self._preserve_known_source_values(
                residual_groups,
                source_groups,
                mask_groups,
            ),
            conditioning_groups,
        )

    def _prepare_flow_target_groups(
        self,
        source_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        target_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        mask_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        emb_groups: Sequence[Optional[Dict[str, Any]]],
        sample_configs: Mapping[int, Dict[str, Any]],
    ) -> List[Optional[Dict[int, torch.Tensor]]]:
        residual_groups, _ = self._prepare_residual_target_and_conditioning(
            source_groups,
            target_groups,
            mask_groups,
            emb_groups,
            sample_configs,
        )
        return residual_groups

    def _prepare_flow_target_and_conditioning(
        self,
        source_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        target_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        mask_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        emb_groups: Sequence[Optional[Dict[str, Any]]],
        sample_configs: Mapping[int, Dict[str, Any]],
    ) -> tuple[
        List[Optional[Dict[int, torch.Tensor]]],
        Optional[List[Optional[Dict[str, Dict[int, torch.Tensor]]]]],
    ]:
        return self._prepare_residual_target_and_conditioning(
            source_groups,
            target_groups,
            mask_groups,
            emb_groups,
            sample_configs,
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
        conditioning_groups: Optional[
            Sequence[Optional[Mapping[str, Mapping[int, torch.Tensor]]]]
        ] = None,
    ) -> List[Optional[Dict[int, torch.Tensor]]]:
        deterministic_groups = (
            [
                None
                if conditioning is None
                else {
                    int(zoom): value
                    for zoom, value in conditioning["mu"].items()
                }
                for conditioning in conditioning_groups
            ]
            if conditioning_groups is not None
            else self._predict_deterministic(
                source_groups,
                mask_groups,
                emb_groups,
                sample_configs,
            )[0]
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
        emb_groups = [
            dict(group) if group else {}
            for group in (
                emb_groups
                if emb_groups is not None
                else [None] * len(source_groups)
            )
        ]

        deterministic_groups, sigma_groups = self._predict_deterministic(
            source_groups,
            mask_groups,
            emb_groups,
            sample_configs,
        )
        conditioning_groups = self._build_spatial_conditioning(
            deterministic_groups,
            sigma_groups,
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
            conditioning_groups=conditioning_groups,
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
