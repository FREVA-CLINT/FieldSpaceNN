from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.grids.grid_layer import GridLayer


def _reduce_to_var_depth(values: torch.Tensor) -> torch.Tensor:
    if values.ndim == 0:
        return values

    reduce_dims = tuple(dim for dim in range(values.ndim) if dim not in (1, values.ndim - 2))
    if not reduce_dims:
        return values

    return values.mean(dim=reduce_dims)


def _expand_mask(mask: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(dim=-1)
    return mask.expand_as(values)


def _masked_reduce_to_var_depth(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return _reduce_to_var_depth(values)

    mask_expanded = _expand_mask(mask, values)
    reduce_dims = tuple(dim for dim in range(values.ndim) if dim not in (1, values.ndim - 2))

    if mask_expanded.dtype == torch.bool:
        weights = mask_expanded.to(dtype=values.dtype)
    else:
        weights = mask_expanded.to(dtype=values.dtype)

    denom = weights.sum(dim=reduce_dims)
    numer = (values * weights).sum(dim=reduce_dims)
    out = torch.zeros_like(numer)
    valid = denom > 0
    out[valid] = numer[valid] / denom[valid]
    return out


def _constant_loss_map(loss: torch.Tensor | float, output: torch.Tensor) -> torch.Tensor:
    scalar = loss if isinstance(loss, torch.Tensor) else torch.tensor(loss, device=output.device, dtype=output.dtype)
    if isinstance(scalar, torch.Tensor) and scalar.ndim > 0:
        scalar = scalar.mean()
    return scalar.expand(output.shape[1], output.shape[-2])


class ReluPressureLevelScaler(nn.Module):
    def __init__(self, minimum: float = 0.2, slope: float = 0.001) -> None:
        super().__init__()
        self.minimum = float(minimum)
        self.slope = float(slope)

    def forward(self, depth_values: torch.Tensor) -> torch.Tensor:
        scaled = depth_values.to(dtype=torch.float32) * self.slope
        return torch.clamp(scaled, min=self.minimum)


class MGMultiLoss(nn.Module):
    def __init__(
        self,
        lambda_dict: Mapping[str, Any],
        grid_layers: Optional[nn.ModuleDict] = None,
        normalize_lambdas: bool = False,
    ):
        super().__init__()
        self.grid_layers: Optional[nn.ModuleDict] = grid_layers
        self.normalize_lambdas: bool = bool(normalize_lambdas)
        self.common_losses: nn.ModuleList = nn.ModuleList()
        self.level_specific_losses: nn.ModuleDict = nn.ModuleDict()

        common_loss_config = lambda_dict.get("common", {})
        for loss_name, lambda_value in common_loss_config.items():
            self._add_loss(loss_name, lambda_value, self.common_losses)

        for key, level_config in lambda_dict.items():
            if str(key).isdigit():
                zoom_level = str(key)
                if zoom_level not in self.level_specific_losses:
                    self.level_specific_losses[zoom_level] = nn.ModuleList()
                for loss_name, lambda_value in level_config.items():
                    self._add_loss(loss_name, lambda_value, self.level_specific_losses[zoom_level], zoom_level)
            elif key != "common" and key in globals():
                self._add_loss(str(key), level_config, self.common_losses)

    def _add_loss(
        self,
        loss_name: str,
        lambda_value: float,
        module_list: nn.ModuleList,
        zoom_level: Optional[str] = None,
    ):
        lambda_val = float(lambda_value)
        if lambda_val <= 0:
            return
        if loss_name not in globals():
            raise KeyError(f"Unknown loss '{loss_name}' in lambda configuration")

        grid_layer = self.grid_layers[zoom_level] if zoom_level is not None and self.grid_layers else None
        loss_instance = globals()[loss_name](grid_layer=grid_layer)
        loss_instance.lambda_val = lambda_val
        loss_instance.loss_name = loss_name
        module_list.append(loss_instance)

    @property
    def has_elements(self):
        return len(self.common_losses) > 0 or any(len(v) > 0 for v in self.level_specific_losses.values())

    def _loss_modules_for_zoom(self, zoom_level: int) -> List[nn.Module]:
        loss_modules = list(self.common_losses)
        if str(zoom_level) in self.level_specific_losses:
            loss_modules.extend(self.level_specific_losses[str(zoom_level)])
        return loss_modules

    def _loss_lambda_normalizer(self, loss_modules: Sequence[nn.Module]) -> float:
        if not self.normalize_lambdas or len(loss_modules) == 0:
            return 1.0

        lambda_sum = sum(float(loss_fcn.lambda_val) for loss_fcn in loss_modules)
        if lambda_sum <= 0:
            return 1.0

        return lambda_sum / float(len(loss_modules))

    def forward(
        self,
        output: Dict[int, torch.Tensor],
        target: Dict[int, torch.Tensor],
        mask: Optional[Dict[int, torch.Tensor]] = None,
        sample_configs: Mapping[int, Dict[str, Any]] = {},
        prefix: str = "",
        emb: Mapping[str, Any] = {},
        variable_weight_map: Optional[torch.Tensor] = None,
        group_index: int = 0,
        group_lambda: float = 1.0,
        normalizer: float = 1.0,
    ):
        loss_dict: Dict[str, float] = {}
        total_loss: Optional[torch.Tensor] = None

        for zoom_level, out_zoom in output.items():
            tgt_zoom = target[zoom_level]
            mask_zoom = mask.get(zoom_level) if mask else None
            sample_conf = sample_configs.get(zoom_level) if sample_configs else None
            loss_modules = self._loss_modules_for_zoom(zoom_level)
            if not loss_modules:
                continue
            loss_lambda_normalizer = self._loss_lambda_normalizer(loss_modules)

            if variable_weight_map is None:
                weight_map = out_zoom.new_ones((out_zoom.shape[1], out_zoom.shape[-2]))
            else:
                weight_map = variable_weight_map.to(device=out_zoom.device, dtype=out_zoom.dtype)
                if weight_map.shape != (out_zoom.shape[1], out_zoom.shape[-2]):
                    raise ValueError(
                        f"variable_weight_map shape {tuple(weight_map.shape)} does not match "
                        f"(variables, depth)=({out_zoom.shape[1]}, {out_zoom.shape[-2]})."
                    )

            var_ids_raw = emb.get("VariableEmbedder", torch.arange(out_zoom.shape[1], device=out_zoom.device))
            if isinstance(var_ids_raw, torch.Tensor) and var_ids_raw.ndim > 1:
                var_ids = var_ids_raw[0]
            elif isinstance(var_ids_raw, torch.Tensor):
                var_ids = var_ids_raw
            else:
                var_ids = torch.arange(out_zoom.shape[1], device=out_zoom.device)

            for loss_fcn in loss_modules:
                if hasattr(loss_fcn, "loss_map"):
                    loss_map = loss_fcn.loss_map(out_zoom, tgt_zoom, mask=mask_zoom, sample_configs=sample_conf)
                else:
                    loss_map = _constant_loss_map(
                        loss_fcn(out_zoom, tgt_zoom, mask=mask_zoom, sample_configs=sample_conf),
                        out_zoom,
                    )

                if not isinstance(loss_map, torch.Tensor):
                    loss_map = _constant_loss_map(loss_map, out_zoom)
                elif loss_map.ndim == 0:
                    loss_map = _constant_loss_map(loss_map, out_zoom)

                weighted_map = (
                    loss_map
                    * weight_map
                    * (float(loss_fcn.lambda_val) / float(loss_lambda_normalizer))
                    * float(group_lambda)
                    / float(normalizer)
                )
                weighted_loss = weighted_map.sum()
                total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss

                aggregate_name = f"{prefix}level{zoom_level}_{loss_fcn._get_name()}"
                loss_dict[aggregate_name] = loss_dict.get(aggregate_name, 0.0) + float(weighted_loss.item())

                for var_pos, var_id in enumerate(var_ids.tolist()):
                    granular_name = (
                        f"{prefix}group{group_index}_level{zoom_level}_var{int(var_id)}_{loss_fcn._get_name()}"
                    )
                    loss_dict[granular_name] = loss_dict.get(granular_name, 0.0) + float(
                        weighted_map[var_pos].sum().item()
                    )

        if total_loss is None:
            first_tensor = next(iter(output.values()), None)
            total_loss = torch.tensor(0.0, device=first_tensor.device if first_tensor is not None else None)

        return total_loss, loss_dict


class L1_loss(nn.Module):
    def __init__(self, **kwargs: Any):
        super().__init__()

    def loss_map(self, output: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return _reduce_to_var_depth(F.smooth_l1_loss(output, target.view(output.shape), reduction="none"))

    def forward(self, output: torch.Tensor, target: torch.Tensor, **kwargs: Any):
        return self.loss_map(output, target, **kwargs).mean()


class MSE_loss(nn.Module):
    def __init__(self, grid_layer: Optional[GridLayer] = None):
        super().__init__()

    def loss_map(self, output: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return _reduce_to_var_depth((output - target.view(output.shape)) ** 2)

    def forward(self, output: torch.Tensor, target: torch.Tensor, **kwargs: Any):
        return self.loss_map(output, target, **kwargs).mean()


class MSE_masked_loss(nn.Module):
    def __init__(self, grid_layer: Optional[GridLayer] = None):
        super().__init__()

    def loss_map(self, output: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        return _masked_reduce_to_var_depth((output - target.view(output.shape)) ** 2, mask)

    def forward(self, output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, **kwargs: Any):
        return self.loss_map(output, target, mask=mask, **kwargs).mean()


class NHInt_loss(nn.Module):
    def __init__(self, grid_layer: GridLayer):
        super().__init__()
        self.grid_layer: GridLayer = grid_layer

    def loss_map(
        self,
        output: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        sample_configs: Dict[str, Any] = {},
        mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        output_nh, _ = self.grid_layer.get_nh(output, **sample_configs)
        target_nh, _ = self.grid_layer.get_nh(target, **sample_configs)
        loss = (output_nh.abs().sum(dim=-2) - target_nh.abs().sum(dim=-2)).abs()
        return _reduce_to_var_depth(loss)

    def forward(self, output: torch.Tensor, target: Optional[torch.Tensor] = None, **kwargs: Any):
        return self.loss_map(output, target, **kwargs).mean()


class NHVar_loss(nn.Module):
    def __init__(self, grid_layer: GridLayer):
        super().__init__()
        self.grid_layer: GridLayer = grid_layer
        self.eps: float = 1e-6

    def loss_map(
        self,
        output: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        sample_configs: Dict[str, Any] = {},
        mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        output_nh, _ = self.grid_layer.get_nh(output, **sample_configs)
        target_nh, _ = self.grid_layer.get_nh(target, **sample_configs)
        out_logstd = 0.5 * torch.log(output_nh.var(dim=-2) + self.eps)
        tgt_logstd = 0.5 * torch.log(target_nh.var(dim=-2) + self.eps)
        return _reduce_to_var_depth((out_logstd - tgt_logstd).abs())

    def forward(self, output: torch.Tensor, target: Optional[torch.Tensor] = None, **kwargs: Any):
        return self.loss_map(output, target, **kwargs).mean()


class GNLL_loss(nn.Module):
    def __init__(self, grid_layer: Optional[GridLayer] = None):
        super().__init__()

    def loss_map(self, output: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        output_mean, output_var = output.chunk(2, dim=-1)
        loss = F.gaussian_nll_loss(output_mean, target.view(*output_mean.shape), output_var, reduction="none")
        return _reduce_to_var_depth(loss)

    def forward(self, output: torch.Tensor, target: torch.Tensor, **kwargs: Any):
        return self.loss_map(output, target, **kwargs).mean()


class MSE_Hole_loss(nn.Module):
    def __init__(self, grid_layer: Optional[GridLayer] = None):
        super().__init__()

    def loss_map(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return _masked_reduce_to_var_depth((output - target.view(output.shape)) ** 2, mask)

    def forward(self, output: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None, **kwargs: Any):
        return self.loss_map(output, target, mask=mask, **kwargs).mean()


class MultiLoss(nn.Module):
    def __init__(self, lambda_dict: Mapping[str, Any], grid_layer: Optional[GridLayer] = None):
        super().__init__()

        self.loss_fcns: List[Dict[str, Any]] = []
        for target, lambda_ in lambda_dict.items():
            if float(lambda_) > 0:
                self.loss_fcns.append(
                    {"lambda": float(lambda_), "fcn": globals()[target](grid_layer=grid_layer)}
                )

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        sample_configs: Dict[str, Any] = {},
        prefix: str = "",
    ):
        loss_dict = {}
        total_loss = 0

        for loss_fcn in self.loss_fcns:
            loss = loss_fcn["fcn"](output, target, mask=mask, sample_configs=sample_configs)
            loss_dict[prefix + loss_fcn["fcn"]._get_name()] = loss.item()
            total_loss = total_loss + loss_fcn["lambda"] * loss

        return total_loss, loss_dict


class Grad_loss(nn.Module):
    def __init__(self, grid_layer: GridLayer):
        super().__init__()
        self.grid_layer: GridLayer = grid_layer
        self.loss_fcn: nn.Module = torch.nn.KLDivLoss(log_target=True, reduction="none")

    def loss_map(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        sample_configs: Dict[str, Any] = {},
        mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        indices_in = (
            sample_configs["indices_layers"][int(self.grid_layer.global_zoom)]
            if sample_configs is not None and isinstance(sample_configs, dict)
            else None
        )
        output_nh, _ = self.grid_layer.get_nh(output, indices_in, sample_configs)
        target_nh, _ = self.grid_layer.get_nh(target.view(output.shape), indices_in, sample_configs)

        nh_diff_output = 1 + (output_nh[:, :, [0]] - output_nh[:, :, 1:]) / output_nh[:, :, [0]]
        nh_diff_target = 1 + (target_nh[:, :, [0]] - target_nh[:, :, 1:]) / target_nh[:, :, [0]]
        nh_diff_output = nh_diff_output.clamp(min=0, max=1)
        nh_diff_target = nh_diff_target.clamp(min=0, max=1)
        nh_diff_output = torch.log_softmax(nh_diff_output, dim=-1)
        nh_diff_target = torch.log_softmax(nh_diff_target, dim=-1)

        kl = self.loss_fcn(nh_diff_output, nh_diff_target)
        return _reduce_to_var_depth(kl)

    def forward(self, output: torch.Tensor, target: torch.Tensor, **kwargs: Any):
        return self.loss_map(output, target, **kwargs).mean()


class NHTV_loss(nn.Module):
    def __init__(self, grid_layer: GridLayer):
        super().__init__()
        self.grid_layer: GridLayer = grid_layer

    def loss_map(
        self,
        output: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        sample_configs: Dict[str, Any] = {},
        mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        output_nh, _ = self.grid_layer.get_nh(output, **sample_configs)
        loss = (output_nh[..., [0], :] - output_nh[..., 1:, :]) ** 2
        return _reduce_to_var_depth(loss).sqrt()

    def forward(self, output: torch.Tensor, target: Optional[torch.Tensor] = None, **kwargs: Any):
        return self.loss_map(output, target, **kwargs).mean()


class NHTV_decay_loss(nn.Module):
    def __init__(self, grid_layer: GridLayer, tau: float = 0.2):
        super().__init__()
        self.grid_layer: GridLayer = grid_layer
        self.tau: float = tau

    def loss_map(
        self,
        output: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        sample_configs: Dict[str, Any] = {},
        mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        output_nh, _ = self.grid_layer.get_nh(output, **sample_configs)
        target_nh, _ = self.grid_layer.get_nh(target, **sample_configs)

        nh_diff = (output_nh[..., [0], :] - output_nh[..., 1:, :]) ** 2
        target_diff = (target_nh[..., [0], :] - target_nh[..., 1:, :]).abs() / (
            target_nh[..., [0], :].abs() + 1e-6
        )
        loss = torch.exp(-target_diff / self.tau) * nh_diff
        return _reduce_to_var_depth(loss).sqrt()

    def forward(self, output: torch.Tensor, target: Optional[torch.Tensor] = None, **kwargs: Any):
        return self.loss_map(output, target, **kwargs).mean()
