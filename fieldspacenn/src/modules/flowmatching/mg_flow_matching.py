import json
import math
import os
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch

from ..grids.grid_utils import encode_zooms, to_zoom


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
        norm_dict: Optional[
            Union[Mapping[Any, Any], str, os.PathLike[str]]
        ] = None,
        temporal_difference: bool = False,
        difference_norm_dict: Optional[
            Union[Mapping[Any, Any], str, os.PathLike[str]]
        ] = None,
    ) -> None:
        """
        Initialize the flow-matching helper.

        :param time_embed_key: Embedding key used to pass continuous flow time.
        :param separate_noise_on_zoom: Whether to sample independent noise per zoom.
        :param interpolation_mode: Flow objective mode. Supported values are
            ``"linear"`` (default) and ``"rectified"``.
        :param rectified_time_epsilon: Lower bound for ``1 - t`` in rectified
            mode for numerical stability near ``t=1``.
        :param norm_dict: Optional per-variable, per-zoom data standard deviations,
            either as a mapping or a path to a JSON file.
        :param temporal_difference: Whether masked targets use normalized one-step
            differences.
        :param difference_norm_dict: Mean/std statistics for temporal differences.
        :return: None.
        """
        self.time_embed_key: str = time_embed_key
        self.separate_noise_on_zoom: bool = separate_noise_on_zoom
        self.norm_dict = self._load_norm_dict(norm_dict, "norm_dict")
        self.temporal_difference: bool = bool(temporal_difference)
        self.difference_norm_dict = self._load_norm_dict(
            difference_norm_dict, "difference_norm_dict"
        )
        if self.temporal_difference:
            if self.difference_norm_dict is None:
                raise ValueError(
                    "`difference_norm_dict` is required when `temporal_difference=true`."
                )
            if not self.separate_noise_on_zoom or self.norm_dict is not None:
                raise ValueError(
                    "Temporal differences require independent, unscaled noise "
                    "(`separate_noise_on_zoom=true`, `norm_dict=null`)."
                )
            self._validate_difference_norm_dict(self.difference_norm_dict)
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
    def _load_norm_dict(
        value: Optional[Union[Mapping[Any, Any], str, os.PathLike[str]]],
        name: str,
    ) -> Optional[Mapping[Any, Any]]:
        if not isinstance(value, (str, os.PathLike)):
            return value
        with open(os.path.expanduser(value), "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, Mapping):
            raise ValueError(f"`{name}` JSON must contain an object at its root.")
        return loaded

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

    @staticmethod
    def _stat_from_definition(definition: Any, statistic: str) -> Optional[Any]:
        if not isinstance(definition, Mapping):
            return definition
        if statistic in definition:
            return definition[statistic]
        stats = definition.get("stats")
        if isinstance(stats, Mapping) and statistic in stats:
            return stats[statistic]
        return None

    @classmethod
    def _validate_difference_norm_dict(cls, config: Mapping[Any, Any]) -> None:
        found_stats = False

        def validate(definition: Any) -> None:
            nonlocal found_stats
            mean = cls._stat_from_definition(definition, "mean")
            std = cls._stat_from_definition(definition, "std")
            if mean is not None or std is not None:
                found_stats = True
                if mean is None or std is None:
                    raise ValueError(
                        "Temporal difference statistics require both mean and std."
                    )
                mean_tensor, std_tensor = torch.as_tensor(mean), torch.as_tensor(std)
                if (
                    not torch.isfinite(mean_tensor).all()
                    or not torch.isfinite(std_tensor).all()
                    or (std_tensor <= 0).any()
                ):
                    raise ValueError(
                        "Temporal difference means must be finite and standard "
                        "deviations positive."
                    )
                return
            if isinstance(definition, Mapping):
                for nested in definition.values():
                    validate(nested)

        validate(config)
        if not found_stats:
            raise ValueError("`difference_norm_dict` contains no mean/std statistics.")

    @staticmethod
    def _variable_names(
        emb_group: Optional[Mapping[str, Any]],
    ) -> Optional[Sequence[Any]]:
        if emb_group is None or "variable_names_sampled" not in emb_group:
            return None
        names = []
        for name in emb_group["variable_names_sampled"]:
            if isinstance(name, str):
                names.append(name)
            elif isinstance(name, Sequence):
                names.append(tuple(str(batch_name) for batch_name in name))
            else:
                names.append(str(name))
        return names

    def _normalization_scale(
        self,
        zoom: int,
        tensor: torch.Tensor,
        variable_names: Optional[Sequence[Any]],
        norm_dict: Optional[Mapping[Any, Any]] = None,
        statistic: str = "std",
        allow_global: bool = False,
    ) -> torch.Tensor:
        """Resolve a normalization statistic and broadcast it to a model tensor."""
        norm_dict = self.norm_dict if norm_dict is None else norm_dict
        assert norm_dict is not None

        # Default collation transposes per-sample name lists into one tuple per
        # variable position. Resolve those names separately for every batch item.
        if variable_names is not None and any(
            not isinstance(name, str) for name in variable_names
        ):
            if len(variable_names) != tensor.shape[1]:
                raise ValueError(
                    "Collated variable names must match the tensor variable axis."
                )
            name_columns = [list(names) for names in variable_names]
            if any(len(names) != tensor.shape[0] for names in name_columns):
                raise ValueError(
                    "Collated variable names must contain one name per batch item."
                )
            batch_scales = [
                self._normalization_scale(
                    zoom,
                    tensor[batch_index:batch_index + 1],
                    [names[batch_index] for names in name_columns],
                    norm_dict,
                    statistic,
                    allow_global,
                )
                for batch_index in range(tensor.shape[0])
            ]
            if all(scale.ndim == 0 for scale in batch_scales):
                if allow_global:
                    return torch.stack(batch_scales).view(-1, 1, 1, 1, 1, 1)
                return batch_scales[0]
            return torch.cat(batch_scales, dim=0)

        zoom_key: Any = zoom if zoom in norm_dict else str(zoom)
        definition = norm_dict.get(zoom_key, norm_dict)
        stat_values = self._stat_from_definition(definition, statistic)
        if stat_values is None:
            names = list(variable_names) if variable_names is not None else list(definition)
            stat_values = []
            for name in names:
                variable_definition = definition[name]
                if isinstance(variable_definition, Mapping):
                    variable_zoom_key: Any = (
                        zoom if zoom in variable_definition else str(zoom)
                    )
                    if not allow_global and zoom_key not in norm_dict:
                        if variable_zoom_key not in variable_definition:
                            raise ValueError(
                                f"Missing zoom {zoom} normalization for variable `{name}`."
                            )
                        variable_definition = variable_definition[variable_zoom_key]
                    else:
                        variable_definition = variable_definition.get(
                            variable_zoom_key, variable_definition
                        )
                stat_values.append(
                    self._stat_from_definition(variable_definition, statistic)
                )

        if stat_values is None or (
            isinstance(stat_values, Sequence)
            and not isinstance(stat_values, (str, bytes))
            and any(value is None for value in stat_values)
        ):
            raise ValueError(f"Could not find normalization {statistic} for zoom {zoom}.")

        value = torch.as_tensor(stat_values, device=tensor.device, dtype=tensor.dtype)
        n_variables, n_depths = tensor.shape[1], tensor.shape[-2]
        if value.ndim == 0:
            return value
        if value.ndim == 1:
            if value.shape[0] == n_variables:
                return value.view(1, n_variables, 1, 1, 1, 1)
            if n_variables == 1 and value.shape[0] == n_depths:
                return value.view(1, 1, 1, 1, n_depths, 1)
        if (
            value.ndim == 2
            and value.shape[0] == n_variables
            and value.shape[1] in {1, n_depths}
        ):
            return value.view(1, n_variables, 1, 1, value.shape[1], 1)
        raise ValueError(
            f"Normalization {statistic} shape {tuple(value.shape)} cannot broadcast to "
            f"zoom {zoom} tensor shape {tuple(tensor.shape)}."
        )

    def encode_temporal_differences(
        self,
        data_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        mask_groups: Optional[Sequence[Optional[Mapping[int, torch.Tensor]]]],
        emb_groups: Optional[Sequence[Optional[Mapping[str, Any]]]] = None,
    ) -> List[Optional[Dict[int, torch.Tensor]]]:
        """Replace masked values after the first timestep with normalized differences."""
        if not self.temporal_difference:
            return [group.copy() if group else None for group in data_groups]
        if mask_groups is None:
            raise ValueError("Temporal differences require masks.")

        emb_groups = emb_groups or [None] * len(data_groups)
        encoded_groups: List[Optional[Dict[int, torch.Tensor]]] = []
        for group, masks, embedding in zip(data_groups, mask_groups, emb_groups):
            if not group:
                encoded_groups.append(group)
                continue
            if not masks:
                raise ValueError("Temporal differences require a mask for every data group.")

            names = self._variable_names(embedding)
            encoded_group: Dict[int, torch.Tensor] = {}
            for zoom, tensor in group.items():
                mean = self._normalization_scale(
                    int(zoom), tensor, names, self.difference_norm_dict, "mean", True
                )
                std = self._normalization_scale(
                    int(zoom), tensor, names, self.difference_norm_dict, "std", True
                )
                encoded = tensor.clone()
                difference = (tensor[:, :, 1:] - tensor[:, :, :-1] - mean) / std
                generated = ~self._known_mask(masks[zoom][:, :, 1:])
                encoded[:, :, 1:] = torch.where(
                    generated.expand_as(difference), difference, tensor[:, :, 1:]
                )
                encoded_group[int(zoom)] = encoded
            encoded_groups.append(encoded_group)
        return encoded_groups

    def decode_temporal_differences(
        self,
        data_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        mask_groups: Optional[Sequence[Optional[Mapping[int, torch.Tensor]]]],
        emb_groups: Optional[Sequence[Optional[Mapping[str, Any]]]] = None,
    ) -> List[Optional[Dict[int, torch.Tensor]]]:
        """Reconstruct absolute masked values from normalized one-step differences."""
        if not self.temporal_difference:
            return [group.copy() if group else None for group in data_groups]
        if mask_groups is None:
            raise ValueError("Temporal differences require masks.")

        emb_groups = emb_groups or [None] * len(data_groups)
        decoded_groups: List[Optional[Dict[int, torch.Tensor]]] = []
        for group, masks, embedding in zip(data_groups, mask_groups, emb_groups):
            if not group:
                decoded_groups.append(group)
                continue
            if not masks:
                raise ValueError("Temporal differences require a mask for every data group.")

            names = self._variable_names(embedding)
            decoded_group: Dict[int, torch.Tensor] = {}
            for zoom, tensor in group.items():
                mean = self._normalization_scale(
                    int(zoom), tensor, names, self.difference_norm_dict, "mean", True
                )
                std = self._normalization_scale(
                    int(zoom), tensor, names, self.difference_norm_dict, "std", True
                )
                steps = [tensor[:, :, :1]]
                for time_index in range(1, tensor.shape[2]):
                    generated = ~self._known_mask(
                        masks[zoom][:, :, time_index:time_index + 1]
                    )
                    reconstructed = (
                        steps[-1]
                        + tensor[:, :, time_index:time_index + 1] * std
                        + mean
                    )
                    steps.append(
                        torch.where(
                            generated.expand_as(reconstructed),
                            reconstructed,
                            tensor[:, :, time_index:time_index + 1],
                        )
                    )
                decoded_group[int(zoom)] = torch.cat(steps, dim=2)
            decoded_groups.append(decoded_group)
        return decoded_groups

    def _generate_normalized_shared_noise(
        self,
        x_zooms: Mapping[int, torch.Tensor],
        zooms: Sequence[int],
        variable_names: Optional[Sequence[Any]],
    ) -> Dict[int, torch.Tensor]:
        """Build unscaled shared pyramid fields, residualize them, then normalize."""
        max_zoom = zooms[0]
        max_time = max(x_zooms[zoom].shape[2] for zoom in zooms)
        finest_shape = list(x_zooms[max_zoom].shape)
        finest_shape[2] = max_time
        finest_noise = torch.randn_like(x_zooms[max_zoom].new_empty(finest_shape))

        raw_zooms: Dict[int, torch.Tensor] = {}
        for zoom in zooms:
            raw_zooms[zoom], _ = to_zoom(
                finest_noise, max_zoom, zoom
            )

        # Use the data pyramid's residual operator on the complete unscaled hierarchy.
        full_field_configs = {
            zoom: {
                "n_past_ts": 0,
                "n_future_ts": 0,
                "zoom_patch_sample": -1,
            }
            for zoom in zooms
        }
        components = encode_zooms(raw_zooms, full_field_configs, {})

        noise_zooms: Dict[int, torch.Tensor] = {}
        for index, zoom in enumerate(zooms):
            time_length = x_zooms[zoom].shape[2]
            component = components[zoom][:, :, -time_length:]
            if component.shape != x_zooms[zoom].shape:
                raise ValueError(
                    f"Cannot share noise at zoom {zoom}: derived shape "
                    f"{tuple(component.shape)} does not match "
                    f"{tuple(x_zooms[zoom].shape)}."
                )
            n_fine = 4 ** (max_zoom - zoom)
            if index == len(zooms) - 1:
                # An average of N independent finest cells has std 1 / sqrt(N).
                raw_std = 1.0 / math.sqrt(n_fine)
            else:
                coarse_zoom = zooms[index + 1]
                n_coarse = 4 ** (max_zoom - coarse_zoom)
                # Var(fine average - containing coarse average) = 1/N_f - 1/N_c.
                raw_std = math.sqrt(1.0 / n_fine - 1.0 / n_coarse)

            norm_std = self._normalization_scale(zoom, component, variable_names)
            noise_zooms[zoom] = component * (norm_std / raw_std)

        return {int(zoom): noise_zooms[int(zoom)] for zoom in x_zooms}

    def generate_noise(
        self,
        x_zooms: Mapping[int, torch.Tensor],
        variable_names: Optional[Sequence[Any]] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Generate Gaussian noise per zoom level.

        :param x_zooms: Input tensors per zoom of shape ``(b, v, t, n, d, f)``.
        :param variable_names: Optional variable names matching the tensor variable axis.
        :return: Noise tensors per zoom with matching shapes.
        """
        if self.separate_noise_on_zoom:
            noise_zooms = {
                int(zoom): torch.randn_like(x_zooms[zoom]) for zoom in x_zooms.keys()
            }
            if self.norm_dict is not None:
                noise_zooms = {
                    int(zoom): noise * self._normalization_scale(
                        int(zoom), noise, variable_names
                    )
                    for zoom, noise in noise_zooms.items()
                }
            return noise_zooms

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
        if self.norm_dict is not None:
            return self._generate_normalized_shared_noise(
                x_zooms, zooms, variable_names
            )

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
        if mask_groups is None:
            mask_groups = [None] * len(gt_groups)
        if emb_groups is None:
            emb_groups = [{} for _ in gt_groups]
        if self.temporal_difference:
            gt_groups = self.encode_temporal_differences(
                gt_groups, mask_groups, emb_groups
            )
        if noise_groups is None:
            noise_groups = [
                self.generate_noise(
                    group,
                    self._variable_names(emb_groups[index])
                    if index < len(emb_groups) else None,
                )
                if group else None
                for index, group in enumerate(gt_groups)
            ]

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
