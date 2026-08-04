from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch


class MGFlowMatching:
    """
    Utilities for Gaussian OT conditional flow-matching training.
    """

    def __init__(
        self,
        time_embed_key: str = "DiffusionStepEmbedder",
        separate_noise_on_zoom: bool = True,
        interpolation_mode: str = "linear",
        rectified_time_epsilon: float = 1e-5,
    ) -> None:
        """
        Initialize the flow-matching helper.

        :param time_embed_key: Embedding key used to pass continuous flow time.
        :param separate_noise_on_zoom: Whether to sample independent noise per zoom.
        :param interpolation_mode: Flow objective mode. Supported values are
            ``"linear"`` (default) and ``"rectified"``.
        :param rectified_time_epsilon: Lower bound for ``1 - t`` in rectified
            mode for numerical stability near ``t=1``.
        :return: None.
        """
        self.time_embed_key: str = time_embed_key
        self.separate_noise_on_zoom: bool = separate_noise_on_zoom
        mode_normalized = str(interpolation_mode).strip().lower()
        if mode_normalized == "recitified":
            mode_normalized = "rectified"
        if mode_normalized not in {"linear", "rectified"}:
            raise ValueError(
                "`interpolation_mode` must be one of {'linear', 'rectified'}, "
                f"got `{interpolation_mode}`."
            )
        self.interpolation_mode: str = mode_normalized
        self.rectified_time_epsilon: float = float(rectified_time_epsilon)
        if self.rectified_time_epsilon <= 0.0:
            raise ValueError("`rectified_time_epsilon` must be > 0.")

    @staticmethod
    def _expand_time_like(times: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
        """
        Broadcast a batch time tensor to a data tensor shape.

        :param times: Tensor of shape ``(b,)``.
        :param tensor: Data tensor with batch dimension first.
        :return: Broadcastable time tensor.
        """
        return times.view(-1, *([1] * (tensor.ndim - 1))).to(device=tensor.device, dtype=tensor.dtype)

    @staticmethod
    def _known_mask(mask: torch.Tensor) -> torch.Tensor:
        """
        Convert the repository mask convention to a known-value mask.

        Diffusion sampling preserves values where the generation mask is False.
        This helper preserves that convention while also tolerating numeric masks.

        :param mask: Boolean or numeric mask tensor.
        :return: Boolean tensor that is True for known/preserved values.
        """
        if mask.dtype == torch.bool:
            return ~mask
        return mask <= 0

    def sample_times(
        self,
        batch_size: int,
        device: torch.device,
        time_range: Tuple[float, float] = (0.0, 1.0),
    ) -> torch.Tensor:
        """
        Uniformly sample continuous flow times.

        :param batch_size: Number of samples.
        :param device: Device for sampled times.
        :param time_range: Inclusive lower/upper time range.
        :return: Tensor of shape ``(batch_size,)`` with values in the range.
        """
        start, end = float(time_range[0]), float(time_range[1])
        if start < 0.0 or end > 1.0 or start > end:
            raise ValueError(f"`time_range` must satisfy 0 <= start <= end <= 1, got {time_range}.")
        return start + (end - start) * torch.rand(batch_size, device=device)

    def generate_noise(self, x_zooms: Mapping[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """
        Generate Gaussian noise per zoom level.

        :param x_zooms: Input tensors per zoom of shape ``(b, v, t, n, d, f)``.
        :return: Noise tensors per zoom with matching shapes.
        """
        if self.separate_noise_on_zoom:
            return {int(zoom): torch.randn_like(x_zooms[zoom]) for zoom in x_zooms.keys()}

        if not x_zooms:
            return {}

        zooms = sorted((int(zoom) for zoom in x_zooms.keys()), reverse=True)
        for zoom in zooms:
            if x_zooms[zoom].ndim != 6:
                raise ValueError(
                    "Shared multizoom noise expects tensors with shape "
                    f"(b, v, t, n, d, f), but zoom {zoom} has shape "
                    f"{tuple(x_zooms[zoom].shape)}."
                )

        max_zoom = zooms[0]
        noise_zooms: Dict[int, torch.Tensor] = {
            max_zoom: torch.randn_like(x_zooms[max_zoom])
        }

        # Build correlated noise recursively so intermediate zooms can provide
        # noise for timesteps that are not present at the finest zoom.
        for higher_zoom, zoom in zip(zooms, zooms[1:]):
            higher = x_zooms[higher_zoom]
            current = x_zooms[zoom]

            compatible_dimensions = (0, 1, 4, 5)
            mismatched_dimensions = [
                dimension
                for dimension in compatible_dimensions
                if higher.shape[dimension] != current.shape[dimension]
            ]
            if mismatched_dimensions:
                raise ValueError(
                    "Cannot share noise between zooms "
                    f"{higher_zoom} and {zoom}: batch, variable, depth, and "
                    "feature dimensions must match, but got shapes "
                    f"{tuple(higher.shape)} and {tuple(current.shape)}."
                )
            if higher.device != current.device:
                raise ValueError(
                    "Cannot share noise between zooms "
                    f"{higher_zoom} and {zoom} on different devices "
                    f"({higher.device} and {current.device})."
                )

            spatial_factor = 4 ** (higher_zoom - zoom)
            expected_higher_size = current.shape[3] * spatial_factor
            if higher.shape[3] != expected_higher_size:
                raise ValueError(
                    "Cannot share noise between zooms "
                    f"{higher_zoom} and {zoom}: expected spatial size "
                    f"{expected_higher_size} at zoom {higher_zoom} "
                    f"(4**{higher_zoom - zoom} times size {current.shape[3]}), "
                    f"but got {higher.shape[3]}."
                )

            sigma = 1.0 / (2 ** (max_zoom - zoom))
            current_noise = torch.randn_like(current) * sigma
            overlap = min(higher.shape[2], current.shape[2])
            if overlap > 0:
                higher_overlap = noise_zooms[higher_zoom][:, :, -overlap:]
                pooled_overlap = higher_overlap.reshape(
                    higher.shape[0],
                    higher.shape[1],
                    overlap,
                    current.shape[3],
                    spatial_factor,
                    higher.shape[4],
                    higher.shape[5],
                ).mean(dim=4)
                current_noise[:, :, -overlap:] = pooled_overlap.to(dtype=current.dtype)

            noise_zooms[zoom] = current_noise

        return {int(zoom): noise_zooms[int(zoom)] for zoom in x_zooms.keys()}

    def apply_mask_to_noise(
        self,
        data_zooms: Mapping[int, torch.Tensor],
        noise_zooms: Mapping[int, torch.Tensor],
        mask_zooms: Optional[Mapping[int, torch.Tensor]],
    ) -> Dict[int, torch.Tensor]:
        """
        Make known regions deterministic by setting their noise endpoint to data.

        :param data_zooms: Data endpoint per zoom.
        :param noise_zooms: Noise endpoint per zoom.
        :param mask_zooms: Optional generation mask per zoom.
        :return: Mask-adjusted noise endpoint per zoom.
        """
        if not mask_zooms:
            return {int(zoom): noise_zooms[zoom] for zoom in noise_zooms.keys()}

        return {
            int(zoom): torch.where(
                self._known_mask(mask_zooms[zoom]),
                data_zooms[zoom],
                noise_zooms[zoom],
            )
            for zoom in data_zooms.keys()
        }

    def interpolate(
        self,
        data_zooms: Mapping[int, torch.Tensor],
        noise_zooms: Mapping[int, torch.Tensor],
        times: torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        """
        Interpolate along the Gaussian OT path.

        :param data_zooms: Data endpoint ``x_1`` per zoom.
        :param noise_zooms: Noise endpoint ``x_0`` per zoom.
        :param times: Continuous times of shape ``(b,)``.
        :return: Interpolated tensors ``x_t = (1 - t) x_0 + t x_1``.
        """
        return {
            int(zoom): (1.0 - self._expand_time_like(times, data_zooms[zoom])) * noise_zooms[zoom]
            + self._expand_time_like(times, data_zooms[zoom]) * data_zooms[zoom]
            for zoom in data_zooms.keys()
        }

    def target_velocity(
        self,
        data_zooms: Mapping[int, torch.Tensor],
        noise_zooms: Mapping[int, torch.Tensor],
        x_t_zooms: Optional[Mapping[int, torch.Tensor]] = None,
        times: Optional[torch.Tensor] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Compute the target velocity for the configured interpolation mode.

        :param data_zooms: Data endpoint ``x_1`` per zoom.
        :param noise_zooms: Noise endpoint ``x_0`` per zoom.
        :param x_t_zooms: Optional interpolated state ``x_t`` per zoom.
        :param times: Optional continuous times of shape ``(b,)``.
        :return: Velocity target per zoom.
        """
        if self.interpolation_mode == "linear":
            return {int(zoom): data_zooms[zoom] - noise_zooms[zoom] for zoom in data_zooms.keys()}

        # Rectified flow with straight interpolation:
        # v* = (x1 - xt) / (1 - t), equivalent to x1 - x0 for exact arithmetic.
        if x_t_zooms is None or times is None:
            return {int(zoom): data_zooms[zoom] - noise_zooms[zoom] for zoom in data_zooms.keys()}

        one_minus_t = (1.0 - times).clamp_min(self.rectified_time_epsilon)
        return {
            int(zoom): (data_zooms[zoom] - x_t_zooms[zoom]) / self._expand_time_like(one_minus_t, data_zooms[zoom])
            for zoom in data_zooms.keys()
        }

    def pred_x1_from_velocity(
        self,
        x_t_zooms: Mapping[int, torch.Tensor],
        velocity_zooms: Mapping[int, torch.Tensor],
        times: torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        """
        Estimate the data endpoint from a velocity prediction.

        :param x_t_zooms: Interpolated state per zoom.
        :param velocity_zooms: Predicted velocity per zoom.
        :param times: Continuous times of shape ``(b,)``.
        :return: Estimated endpoint ``x_1`` per zoom.
        """
        return {
            int(zoom): x_t_zooms[zoom]
            + (1.0 - self._expand_time_like(times, x_t_zooms[zoom])) * velocity_zooms[zoom]
            for zoom in x_t_zooms.keys()
        }

    def _with_time_embedding(
        self,
        emb_groups: Optional[Sequence[Optional[Dict[str, Any]]]],
        n_groups: int,
        times: torch.Tensor,
    ) -> List[Dict[str, Any]]:
        """
        Return embedding groups with the flow time injected.

        :param emb_groups: Optional original embedding groups.
        :param n_groups: Number of data groups.
        :param times: Continuous times of shape ``(b,)``.
        :return: Embedding groups containing ``self.time_embed_key``.
        """
        if emb_groups is None:
            emb_groups = [{} for _ in range(n_groups)]

        output: List[Dict[str, Any]] = []
        for emb in emb_groups:
            emb_out = dict(emb) if emb else {}
            emb_out[self.time_embed_key] = times
            output.append(emb_out)
        return output

    def model_velocity(
        self,
        model: Callable,
        x_t_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        times: torch.Tensor,
        mask_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        emb_groups: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
        **model_kwargs: Any,
    ) -> Sequence[Optional[Dict[int, torch.Tensor]]]:
        """
        Evaluate the velocity model at a continuous flow time.

        :param model: Callable velocity model.
        :param x_t_groups: Current state groups.
        :param times: Continuous times of shape ``(b,)``.
        :param mask_groups: Optional mask groups.
        :param emb_groups: Optional embedding groups.
        :param model_kwargs: Additional model keyword arguments.
        :return: Predicted velocity groups.
        """
        emb_groups_with_time = self._with_time_embedding(emb_groups, len(x_t_groups), times)
        return model(
            x_t_groups,
            emb_groups=emb_groups_with_time,
            mask_zooms_groups=mask_groups,
            **model_kwargs,
        )

    def training_losses(
        self,
        model: Callable,
        gt_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        times: torch.Tensor,
        mask_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        emb_groups: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
        noise_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        create_pred_x1: bool = False,
        **model_kwargs: Any,
    ) -> List[Tuple[Optional[Dict[int, torch.Tensor]], Optional[Dict[int, torch.Tensor]], Optional[Dict[int, torch.Tensor]]]]:
        """
        Compute per-group flow-matching training targets and model outputs.

        :param model: Velocity model.
        :param gt_groups: Data endpoint groups.
        :param times: Continuous times of shape ``(b,)``.
        :param mask_groups: Optional mask groups.
        :param emb_groups: Optional embedding groups.
        :param noise_groups: Optional pre-sampled noise endpoint groups.
        :param create_pred_x1: Whether to return estimated data endpoints.
        :param model_kwargs: Additional model keyword arguments.
        :return: List of ``(target_velocity, model_output, pred_x1)`` tuples.
        """
        if noise_groups is None:
            noise_groups = [self.generate_noise(group) if group else None for group in gt_groups]

        if mask_groups is None:
            mask_groups = [None] * len(gt_groups)
        if emb_groups is None:
            emb_groups = [{} for _ in gt_groups]

        adjusted_noise_groups: List[Optional[Dict[int, torch.Tensor]]] = []
        x_t_groups: List[Optional[Dict[int, torch.Tensor]]] = []
        target_groups: List[Optional[Dict[int, torch.Tensor]]] = []

        for gt_zooms, noise_zooms, mask_zooms in zip(gt_groups, noise_groups, mask_groups):
            if not gt_zooms:
                adjusted_noise_groups.append(None)
                x_t_groups.append(None)
                target_groups.append(None)
                continue

            assert noise_zooms is not None
            adjusted_noise = self.apply_mask_to_noise(gt_zooms, noise_zooms, mask_zooms)
            adjusted_noise_groups.append(adjusted_noise)
            x_t_zooms = self.interpolate(gt_zooms, adjusted_noise, times)
            x_t_groups.append(x_t_zooms)
            target_groups.append(
                self.target_velocity(gt_zooms, adjusted_noise, x_t_zooms=x_t_zooms, times=times)
            )

        model_output_groups = list(
            self.model_velocity(
                model,
                x_t_groups,
                times,
                mask_groups=mask_groups,
                emb_groups=emb_groups,
                **model_kwargs,
            )
        )

        pred_x1_groups: List[Optional[Dict[int, torch.Tensor]]] = []
        for idx, (x_t_zooms, target_zooms, model_output, mask_zooms) in enumerate(
            zip(x_t_groups, target_groups, model_output_groups, mask_groups)
        ):
            if not x_t_zooms or not target_zooms or not model_output:
                pred_x1_groups.append(None)
                continue

            if mask_zooms:
                model_output_groups[idx] = {
                    int(zoom): torch.where(
                        self._known_mask(mask_zooms[zoom]),
                        target_zooms[zoom],
                        model_output[zoom],
                    )
                    for zoom in target_zooms.keys()
                }
                model_output = model_output_groups[idx]

            if create_pred_x1:
                pred_x1_groups.append(self.pred_x1_from_velocity(x_t_zooms, model_output, times))
            else:
                pred_x1_groups.append(None)

        return list(zip(target_groups, model_output_groups, pred_x1_groups))
