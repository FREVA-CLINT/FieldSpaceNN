"""CRA5-style rate-distortion loss adapted to FieldSpace HEALPix tensors."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ...utils.losses import MGMultiLoss
from .entropy_adapters import flatten_likelihoods_for_bpp


class HEALPixCRA5RateDistortionLoss(nn.Module):
    """Rate-distortion-KL objective for hyper-prior FieldSpace autoencoders."""

    def __init__(
        self,
        reconstruction_loss_config: Optional[Mapping[str, Any]] = None,
        lmbda: float = 1.0,
        bpp_weight: float = 1.0,
        kl_weight: float = 1e-6,
        aux_weight: float = 0.0,
        original_bits_per_value: int = 32,
        rate_normalization: str = "spatiotemporal_cells",
        use_learned_log_variance: bool = False,
        logvar_init: float = 0.0,
        variable_weights: Optional[Sequence[float]] = None,
        eps: float = 1e-9,
        grid_layers: Optional[nn.ModuleDict] = None,
    ) -> None:
        super().__init__()
        self.lmbda = float(lmbda)
        self.bpp_weight = float(bpp_weight)
        self.kl_weight = float(kl_weight)
        self.aux_weight = float(aux_weight)
        self.original_bits_per_value = int(original_bits_per_value)
        self.rate_normalization = str(rate_normalization)
        self.variable_weights = variable_weights
        self.eps = float(eps)
        self.reconstruction_loss_config = reconstruction_loss_config
        self.reconstruction_loss: Optional[MGMultiLoss] = None
        if reconstruction_loss_config is not None:
            zooms_config = reconstruction_loss_config.get("zooms", reconstruction_loss_config)
            self.reconstruction_loss = MGMultiLoss(zooms_config, grid_layers=grid_layers)
        if use_learned_log_variance:
            self.logvar = nn.Parameter(torch.tensor(float(logvar_init)))
        else:
            self.register_buffer("logvar", torch.tensor(float(logvar_init)), persistent=False)

    def forward(
        self,
        output: Mapping[str, Any],
        target_groups: Sequence[Optional[Mapping[int, torch.Tensor]]],
        mask_groups: Optional[Sequence[Optional[Mapping[int, torch.Tensor]]]] = None,
        sample_configs: Mapping[int, Any] = {},
        emb_groups: Optional[Sequence[Mapping[str, Any]]] = None,
        model: Optional[nn.Module] = None,
        prefix: str = "train",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute total loss and diagnostics."""

        x_hat_groups = output["x_hat"]
        rec_loss = self._reconstruction_loss(x_hat_groups, target_groups, mask_groups, sample_configs, emb_groups)
        rec_loss = rec_loss / torch.exp(self.logvar) + self.logvar

        posterior = output.get("posterior")
        kl_loss = posterior.kl(reduction="mean") if posterior is not None else rec_loss.new_tensor(0.0)
        if not torch.is_tensor(kl_loss):
            kl_loss = rec_loss.new_tensor(float(kl_loss))
        kl_loss = kl_loss.to(device=rec_loss.device)

        bits_y = self._likelihood_bits(output.get("likelihoods", {}).get("y"), rec_loss.device)
        bits_z = self._likelihood_bits(output.get("likelihoods", {}).get("z"), rec_loss.device)
        side_bits = self._passthrough_bits(output.get("passthrough"), rec_loss.device)
        estimated_bits = bits_y + bits_z + side_bits
        denominator = self._rate_denominator(target_groups, rec_loss.device)
        bpp_y = bits_y / denominator.clamp_min(self.eps)
        bpp_z = bits_z / denominator.clamp_min(self.eps)
        bpp_side = side_bits / denominator.clamp_min(self.eps)
        bpp_loss = self.bpp_weight * (bpp_y + bpp_z + bpp_side)

        aux_loss = rec_loss.new_tensor(0.0)
        if model is not None and self.aux_weight > 0.0 and hasattr(model, "aux_loss"):
            aux_loss = model.aux_loss().to(device=rec_loss.device)

        total_loss = self.lmbda * rec_loss + self.kl_weight * kl_loss + bpp_loss + self.aux_weight * aux_loss
        original_bits = self._original_bits(target_groups, rec_loss.device)
        compression_ratio = original_bits / estimated_bits.clamp_min(self.eps)

        loss_dict = {
            f"{prefix}/total_loss": total_loss.detach(),
            f"{prefix}/rec_loss": rec_loss.detach(),
            f"{prefix}/kl_loss": kl_loss.detach(),
            f"{prefix}/bpp_loss": bpp_loss.detach(),
            f"{prefix}/bpp_y": bpp_y.detach(),
            f"{prefix}/bpp_z": bpp_z.detach(),
            f"{prefix}/bpp_side": bpp_side.detach(),
            f"{prefix}/side_bits": side_bits.detach(),
            f"{prefix}/estimated_bits": estimated_bits.detach(),
            f"{prefix}/estimated_compression_ratio": compression_ratio.detach(),
            f"{prefix}/aux_loss": aux_loss.detach(),
        }
        return total_loss, loss_dict

    def _reconstruction_loss(
        self,
        x_hat_groups: Sequence[Optional[Mapping[int, torch.Tensor]]],
        target_groups: Sequence[Optional[Mapping[int, torch.Tensor]]],
        mask_groups: Optional[Sequence[Optional[Mapping[int, torch.Tensor]]]],
        sample_configs: Mapping[int, Any],
        emb_groups: Optional[Sequence[Mapping[str, Any]]],
    ) -> torch.Tensor:
        if self.reconstruction_loss is not None:
            total: Optional[torch.Tensor] = None
            count = 0
            for group_idx, (x_hat, target) in enumerate(zip(x_hat_groups, target_groups)):
                if not x_hat:
                    continue
                mask = mask_groups[group_idx] if mask_groups is not None else None
                emb = emb_groups[group_idx] if emb_groups is not None else {}
                loss, _ = self.reconstruction_loss(
                    dict(x_hat),
                    dict(target),
                    mask=dict(mask) if mask else None,
                    sample_configs=sample_configs,
                    emb=emb,
                )
                total = loss if total is None else total + loss
                count += 1
            if total is not None:
                return total / max(count, 1)
        return self._default_masked_mse(x_hat_groups, target_groups, mask_groups)

    def _default_masked_mse(
        self,
        x_hat_groups: Sequence[Optional[Mapping[int, torch.Tensor]]],
        target_groups: Sequence[Optional[Mapping[int, torch.Tensor]]],
        mask_groups: Optional[Sequence[Optional[Mapping[int, torch.Tensor]]]],
    ) -> torch.Tensor:
        total: Optional[torch.Tensor] = None
        weight_total: Optional[torch.Tensor] = None
        for group_idx, (x_hat_group, target_group) in enumerate(zip(x_hat_groups, target_groups)):
            if not x_hat_group:
                continue
            mask_group = mask_groups[group_idx] if mask_groups is not None else None
            for zoom, x_hat in x_hat_group.items():
                target = target_group[int(zoom)].view_as(x_hat)
                residual = (x_hat - target).pow(2)
                residual = self._apply_variable_weights(residual)
                if mask_group is not None and int(zoom) in mask_group:
                    mask = mask_group[int(zoom)].view_as(x_hat)
                    if mask.dtype == torch.bool:
                        residual = residual.masked_select(mask)
                        weight = residual.new_tensor(float(residual.numel()))
                        value = residual.sum()
                    else:
                        weighted = residual * mask
                        value = weighted.sum()
                        weight = mask.sum().to(dtype=weighted.dtype).clamp_min(self.eps)
                else:
                    value = residual.sum()
                    weight = residual.new_tensor(float(residual.numel()))
                total = value if total is None else total + value
                weight_total = weight if weight_total is None else weight_total + weight
        if total is None or weight_total is None:
            raise ValueError("No overlapping tensors found for reconstruction loss.")
        return total / weight_total.clamp_min(self.eps)

    def _apply_variable_weights(self, residual: torch.Tensor) -> torch.Tensor:
        if self.variable_weights is None:
            return residual
        weights = torch.as_tensor(self.variable_weights, device=residual.device, dtype=residual.dtype)
        if weights.numel() != residual.shape[1]:
            raise ValueError(
                f"variable_weights length {weights.numel()} does not match variable dimension {residual.shape[1]}."
            )
        return residual * weights.view(1, -1, 1, 1, 1, 1)

    def _likelihood_bits(
        self,
        likelihoods: Optional[Sequence[Optional[Mapping[int, torch.Tensor]]]],
        device: torch.device,
    ) -> torch.Tensor:
        bits = torch.tensor(0.0, device=device)
        for likelihood in flatten_likelihoods_for_bpp(likelihoods):
            bits = bits - torch.log2(likelihood.to(device=device).clamp(self.eps, 1.0)).sum()
        return bits

    def _passthrough_bits(
        self,
        passthrough_groups: Optional[Sequence[Optional[Mapping[int, torch.Tensor]]]],
        device: torch.device,
    ) -> torch.Tensor:
        if passthrough_groups is None:
            return torch.tensor(0.0, device=device)
        numel = 0
        for group in passthrough_groups:
            if not group:
                continue
            numel += sum(int(tensor.numel()) for tensor in group.values())
        return torch.tensor(float(numel * self.original_bits_per_value), device=device)

    def _rate_denominator(
        self,
        target_groups: Sequence[Optional[Mapping[int, torch.Tensor]]],
        device: torch.device,
    ) -> torch.Tensor:
        value = 0
        for group in target_groups:
            if not group:
                continue
            for tensor in group.values():
                if self.rate_normalization == "spatiotemporal_cells":
                    value += int(tensor.shape[0] * tensor.shape[2] * tensor.shape[3] * tensor.shape[4])
                elif self.rate_normalization == "values":
                    value += int(tensor.numel())
                elif self.rate_normalization == "batch":
                    value += int(tensor.shape[0])
                else:
                    raise ValueError(f"Unknown rate_normalization '{self.rate_normalization}'.")
        return torch.tensor(float(value), device=device)

    def _original_bits(
        self,
        target_groups: Sequence[Optional[Mapping[int, torch.Tensor]]],
        device: torch.device,
    ) -> torch.Tensor:
        numel = 0
        for group in target_groups:
            if not group:
                continue
            numel += sum(int(tensor.numel()) for tensor in group.values())
        return torch.tensor(float(numel * self.original_bits_per_value), device=device)
