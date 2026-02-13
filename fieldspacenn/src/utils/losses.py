from typing import Any, Dict, List, Mapping, Optional

import torch
import torch.nn as nn

from ..modules.grids.grid_layer import GridLayer


class MGMultiLoss(nn.Module):
    def __init__(
        self,
        lambda_dict: Mapping[str, Any],
        grid_layers: Optional[nn.ModuleDict] = None
    ):
        """
        Configure a composite loss across common and per-zoom components.

        :param lambda_dict: Dictionary with loss weights and zoom-specific configs.
        :param grid_layers: Optional grid layers used by spatial losses.
        :return: None.
        """
        super().__init__()
        self.grid_layers: Optional[nn.ModuleDict] = grid_layers
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

    def _add_loss(
        self,
        loss_name: str,
        lambda_value: float,
        module_list: nn.ModuleList,
        zoom_level: Optional[str] = None,
    ):
        """
        Instantiate and register a loss module with its lambda weight.

        :param loss_name: Name of the loss function (must exist in globals).
        :param lambda_value: Scalar weight for the loss.
        :param module_list: ModuleList to append the instantiated loss to.
        :param zoom_level: Optional zoom level for zoom-specific losses.
        :return: None.
        """
        lambda_val = float(lambda_value)
        if lambda_val > 0:
            if loss_name not in globals():
                raise KeyError(f"Unknown loss '{loss_name}' in lambda configuration")

            grid_layer = self.grid_layers[zoom_level] if zoom_level is not None and self.grid_layers else None
            loss_instance = globals()[loss_name](grid_layer=grid_layer)
            
            # Store lambda and name on the instance for easy access in forward
            loss_instance.lambda_val = lambda_val
            loss_instance.loss_name = loss_name
            module_list.append(loss_instance)

    @property
    def has_elements(self):
        return len(self.common_losses) > 0 or any(len(v) > 0 for v in self.level_specific_losses.values())

    def forward(
        self,
        output: Dict[int, torch.Tensor],
        target: Dict[int, torch.Tensor],
        mask: Optional[Dict[int, torch.Tensor]] = None,
        sample_configs: Mapping[int, Dict[str, Any]] = {},
        prefix: str = '',
        emb: Mapping[str, Any] = {},
    ):
        """
        Compute the weighted loss across zoom levels.

        :param output: Output mapping per zoom with tensors of shape ``(b, v, t, n, d, f)``.
        :param target: Target mapping per zoom with tensors of shape ``(b, v, t, n, d, f)``.
        :param mask: Optional mask mapping per zoom aligned with ``output``.
        :param sample_configs: Sampling configuration dictionary per zoom.
        :param prefix: Prefix for loss names in the returned dictionary.
        :param emb: Optional embedding dictionary passed to losses that need it.
        :return: Tuple of ``(total_loss, loss_dict)``.
        """
        loss_dict = {}
        total_loss = 0.0

        for zoom_level, out_zoom in output.items():
            tgt_zoom = target[zoom_level]
            mask_zoom = mask.get(zoom_level) if mask else None
            sample_conf = sample_configs.get(zoom_level) if sample_configs else None
            
            # Apply common losses
            for loss_fcn in self.common_losses:
                loss = loss_fcn(out_zoom, tgt_zoom, mask=mask_zoom, sample_configs=sample_conf)
                name = f"{prefix}level{zoom_level}_{loss_fcn._get_name()}"
                if isinstance(loss, torch.Tensor):
                    loss_dict[name] = loss.item()
                    total_loss += loss_fcn.lambda_val * loss
                else:
                    loss_dict[name] = loss

            # Apply level-specific losses
            if str(zoom_level) in self.level_specific_losses:
                for loss_fcn in self.level_specific_losses[str(zoom_level)]:
                    loss = loss_fcn(out_zoom, tgt_zoom, mask=mask_zoom, sample_configs=sample_conf)
                    name = f"{prefix}level{zoom_level}_{loss_fcn._get_name()}"
                    if isinstance(loss, torch.Tensor):
                        loss_dict[name] = loss.item()
                        total_loss += loss_fcn.lambda_val * loss
                    else:
                        loss_dict[name] = loss
        return total_loss, loss_dict


class L1_loss(nn.Module):
    """
    Smooth L1 loss for dense outputs.
    """
    def __init__(self, **kwargs: Any):
        super().__init__()

        self.loss_fcn: nn.Module = torch.nn.SmoothL1Loss()

    def forward(self, output: torch.Tensor, target: torch.Tensor, **kwargs: Any):
        """
        Compute Smooth L1 loss.

        :param output: Output tensor of shape ``(b, v, t, n, d, f)``.
        :param target: Target tensor of shape ``(b, v, t, n, d, f)``.
        :param kwargs: Additional keyword arguments (unused).
        :return: Loss scalar tensor.
        """
        loss = self.loss_fcn(output, target.view(output.shape))
        return loss

class MSE_loss(nn.Module):
    """
    Mean-squared error loss for dense outputs.
    """
    def __init__(self, grid_layer: Optional[GridLayer] = None):
        super().__init__()

        self.loss_fcn: nn.Module = torch.nn.MSELoss() 

    def forward(self, output: torch.Tensor, target: torch.Tensor, **kwargs: Any):
        """
        Compute MSE loss.

        :param output: Output tensor of shape ``(b, v, t, n, d, f)``.
        :param target: Target tensor of shape ``(b, v, t, n, d, f)``.
        :param kwargs: Additional keyword arguments (unused).
        :return: Loss scalar tensor.
        """
        loss = self.loss_fcn(output, target.view(output.shape))
        return loss
    
class MSE_masked_loss(nn.Module):
    """
    Mean-squared error loss computed on masked elements only.
    """
    def __init__(self, grid_layer: Optional[GridLayer] = None):
        super().__init__()

        self.loss_fcn: nn.Module = torch.nn.MSELoss() 

    def forward(self, output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, **kwargs: Any):
        """
        Compute MSE loss on masked elements.

        :param output: Output tensor of shape ``(b, v, t, n, d, f)``.
        :param target: Target tensor of shape ``(b, v, t, n, d, f)``.
        :param mask: Boolean mask tensor aligned with output.
        :param kwargs: Additional keyword arguments (unused).
        :return: Loss scalar tensor.
        """
        if mask.any():
            loss = self.loss_fcn(output[mask], target.view(output.shape)[mask])
        else:            
            loss = 0.0
        return loss
    
class NHInt_loss(nn.Module):
    """
    Neighborhood integral matching loss across grid neighborhoods.
    """
    def __init__(self, grid_layer: GridLayer):
        super().__init__()
        self.grid_layer: GridLayer = grid_layer
        self.eps: float = 1e-6

    def forward(self, output: torch.Tensor, target: Optional[torch.Tensor] = None, sample_configs: Dict[str, Any] = {}, mask: Optional[torch.Tensor] = None, **kwargs: Any):
        """
        Compute neighborhood integral loss.

        :param output: Output tensor of shape ``(b, v, t, n, d, f)``.
        :param target: Target tensor of shape ``(b, v, t, n, d, f)``.
        :param sample_configs: Sampling configuration for neighborhood gathering.
        :param mask: Optional mask tensor (unused).
        :param kwargs: Additional keyword arguments (unused).
        :return: Loss scalar tensor.
        """
        output_nh, _ = self.grid_layer.get_nh(output, **sample_configs)

        target_nh, _ = self.grid_layer.get_nh(target, **sample_configs)
    
        loss = (output_nh.abs().sum(dim=-2) - target_nh.abs().sum(dim=-2)).abs().mean()

        return loss
    
class NHVar_loss(nn.Module):
    """
    Neighborhood variance loss comparing log-variance between output and target.
    """
    def __init__(self, grid_layer: GridLayer):
        super().__init__()
        self.grid_layer: GridLayer = grid_layer
        self.eps: float = 1e-6

    def forward(self, output: torch.Tensor, target: Optional[torch.Tensor] = None, sample_configs: Dict[str, Any] = {}, mask: Optional[torch.Tensor] = None, **kwargs: Any):
        """
        Compute neighborhood variance loss.

        :param output: Output tensor of shape ``(b, v, t, n, d, f)``.
        :param target: Target tensor of shape ``(b, v, t, n, d, f)``.
        :param sample_configs: Sampling configuration for neighborhood gathering.
        :param mask: Optional mask tensor (unused).
        :param kwargs: Additional keyword arguments (unused).
        :return: Loss scalar tensor.
        """
        output_nh, _ = self.grid_layer.get_nh(output, **sample_configs)

        target_nh, _ = self.grid_layer.get_nh(target, **sample_configs)
        
        out_logstd = 0.5 * torch.log((output_nh).var(dim=-2) + self.eps)
        tgt_logstd = 0.5 * torch.log((target_nh).var(dim=-2) + self.eps)

        loss = (out_logstd - tgt_logstd).abs().mean()

        return loss
    
class GNLL_loss(nn.Module):
    """
    Gaussian negative log-likelihood loss.
    """
    def __init__(self, grid_layer: Optional[GridLayer] = None):
        super().__init__()

        self.loss_fcn: nn.Module = torch.nn.GaussianNLLLoss() 

    def forward(self, output: torch.Tensor, target: torch.Tensor, **kwargs: Any):
        """
        Compute Gaussian NLL loss.

        :param output: Output tensor with mean and variance concatenated on last dim.
        :param target: Target tensor of shape ``(b, v, t, n, d, f)``.
        :param kwargs: Additional keyword arguments (unused).
        :return: Loss scalar tensor.
        """
        output, output_var = output.chunk(2, dim=-1)
        loss = self.loss_fcn(output, target.view(*output.shape), output_var)
        return loss
    
class MSE_Hole_loss(nn.Module):
    """
    Mean-squared error loss that optionally ignores masked elements.
    """
    def __init__(self, grid_layer: Optional[GridLayer] = None):
        super().__init__()

        self.loss_fcn: nn.Module = torch.nn.MSELoss() 

    def forward(self, output: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None, **kwargs: Any):
        """
        Compute MSE loss with optional masking.

        :param output: Output tensor of shape ``(b, v, t, n, d, f)``.
        :param target: Target tensor of shape ``(b, v, t, n, d, f)``.
        :param mask: Optional mask tensor aligned with output.
        :param kwargs: Additional keyword arguments (unused).
        :return: Loss scalar tensor.
        """
        if mask is not None:
            if mask.dtype != torch.bool:
                target = target.view(output.shape) * mask
                output = output * mask
            else:
                mask = mask.view(output.shape)
                target = target.view(output.shape)[mask.expand_as(output)]
                output = output[mask.expand_as(output)]
        loss = self.loss_fcn(output, target.view(output.shape))
        return loss

class MultiLoss(nn.Module):
    """
    Simple weighted multi-loss wrapper for a single zoom.
    """
    def __init__(self, lambda_dict: Mapping[str, Any], grid_layer: Optional[GridLayer] = None):
        super().__init__()

        self.loss_fcns: List[Dict[str, Any]] = []
        for target, lambda_ in lambda_dict.items():
            if float(lambda_) > 0:
                self.loss_fcns.append({'lambda': float(lambda_), 
                                        'fcn': globals()[target](grid_layer=grid_layer)})
    
    def forward(self, output: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None, sample_configs: Dict[str, Any] = {}, prefix: str = ''):
        """
        Compute weighted loss for a single zoom.

        :param output: Output tensor of shape ``(b, v, t, n, d, f)``.
        :param target: Target tensor of shape ``(b, v, t, n, d, f)``.
        :param mask: Optional mask tensor.
        :param sample_configs: Sampling configuration for neighborhood losses.
        :param prefix: Prefix for loss keys.
        :return: Tuple of (total_loss, loss_dict).
        """
        loss_dict = {}
        total_loss = 0

        for loss_fcn in self.loss_fcns:
            loss = loss_fcn['fcn'](output, target, mask=mask, sample_configs=sample_configs)
            loss_dict[prefix + loss_fcn['fcn']._get_name()] = loss.item()
            total_loss = total_loss + loss_fcn['lambda'] * loss
        
        return total_loss, loss_dict


class Grad_loss(nn.Module):
    """
    Gradient-based neighborhood loss using KL divergence on normalized diffs.
    """
    def __init__(self, grid_layer: GridLayer):
        super().__init__()
        self.grid_layer: GridLayer = grid_layer
        self.loss_fcn: nn.Module = torch.nn.KLDivLoss(log_target=True, reduction='batchmean') 

    def forward(self, output: torch.Tensor, target: torch.Tensor, sample_configs: Dict[str, Any] = {}, mask: Optional[torch.Tensor] = None, **kwargs: Any):
        """
        Compute gradient-based KL divergence loss over neighborhoods.

        :param output: Output tensor of shape ``(b, v, t, n, d, f)``.
        :param target: Target tensor of shape ``(b, v, t, n, d, f)``.
        :param sample_configs: Sampling configuration with neighbor indices.
        :param mask: Optional mask tensor (unused).
        :param kwargs: Additional keyword arguments (unused).
        :return: Loss scalar tensor.
        """
        indices_in  = sample_configs["indices_layers"][int(self.grid_layer.global_zoom)] if sample_configs is not None and isinstance(sample_configs, dict) else None
        output_nh, _ = self.grid_layer.get_nh(output, indices_in, sample_configs)
        
        target_nh, _ = self.grid_layer.get_nh(target.view(output.shape), indices_in, sample_configs)

        # Compute normalized neighbor differences and compare distributions via KL.
        nh_diff_output = 1+(output_nh[:,:,[0]] - output_nh[:,:,1:])/output_nh[:,:,[0]]
        nh_diff_target = 1+(target_nh[:,:,[0]] - target_nh[:,:,1:])/target_nh[:,:,[0]]

        nh_diff_output = nh_diff_output.view(output_nh.shape[0],-1).clamp(min=0, max=1)
        nh_diff_target = nh_diff_target.view(output_nh.shape[0],-1).clamp(min=0, max=1)

        nh_diff_output = nn.functional.log_softmax(nh_diff_output, dim=-1)
        nh_diff_target = nn.functional.log_softmax(nh_diff_target, dim=-1)

        loss = self.loss_fcn(nh_diff_output, nh_diff_target)
        return loss

class NHTV_loss(nn.Module):
    """
    Neighborhood total variation loss.
    """
    def __init__(self, grid_layer: GridLayer):
        super().__init__()
        self.grid_layer: GridLayer = grid_layer

    def forward(self, output: torch.Tensor, target: Optional[torch.Tensor] = None, sample_configs: Dict[str, Any] = {}, mask: Optional[torch.Tensor] = None, **kwargs: Any):
        """
        Compute neighborhood total variation loss.

        :param output: Output tensor of shape ``(b, v, t, n, d, f)``.
        :param target: Target tensor (unused).
        :param sample_configs: Sampling configuration for neighborhood gathering.
        :param mask: Optional mask tensor (unused).
        :param kwargs: Additional keyword arguments (unused).
        :return: Loss scalar tensor.
        """
        output_nh, _ = self.grid_layer.get_nh(output, **sample_configs)


        loss = (((output_nh[...,[0],:] - output_nh[...,1:,:]))**2).mean().sqrt()
        return loss
    

class NHTV_decay_loss(nn.Module):
    """
    Neighborhood total variation loss with target-driven decay.
    """
    def __init__(self, grid_layer: GridLayer, tau: float = 0.2):
        super().__init__()
        self.grid_layer: GridLayer = grid_layer
        self.tau: float = tau

    def forward(self, output: torch.Tensor, target: Optional[torch.Tensor] = None, sample_configs: Dict[str, Any] = {}, mask: Optional[torch.Tensor] = None, **kwargs: Any):
        """
        Compute decayed neighborhood total variation loss.

        :param output: Output tensor of shape ``(b, v, t, n, d, f)``.
        :param target: Target tensor of shape ``(b, v, t, n, d, f)``.
        :param sample_configs: Sampling configuration for neighborhood gathering.
        :param mask: Optional mask tensor (unused).
        :param kwargs: Additional keyword arguments (unused).
        :return: Loss scalar tensor.
        """
        output_nh, _ = self.grid_layer.get_nh(output, **sample_configs)
        target_nh, _ = self.grid_layer.get_nh(target, **sample_configs)

        nh_diff = (output_nh[...,[0],:] - output_nh[...,1:,:])**2
        target_diff = (target_nh[...,[0],:] - target_nh[...,1:,:]).abs()/(target_nh[...,[0],:].abs()+1e-6)

        loss = (torch.exp(-1*target_diff/self.tau)*nh_diff).mean().sqrt()

        return loss
