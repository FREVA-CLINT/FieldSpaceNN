import torch
import torch.nn as nn

from ..modules.grids.grid_layer import GridLayer

class L1_loss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.loss_fcn = torch.nn.SmoothL1Loss()

    def forward(self, output, target, **kwargs):
        loss = self.loss_fcn(output, target.view(output.shape))
        return loss

class MSE_loss(nn.Module):
    def __init__(self, grid_layer=None):
        super().__init__()

        self.loss_fcn = torch.nn.MSELoss() 

    def forward(self, output, target, **kwargs):
        loss = self.loss_fcn(output, target.view(output.shape))
        return loss
    
class MSE_masked_loss(nn.Module):
    def __init__(self, grid_layer=None):
        super().__init__()

        self.loss_fcn = torch.nn.MSELoss() 

    def forward(self, output, target, mask, **kwargs):
        loss = self.loss_fcn(output, target.view(output.shape))
        return loss
    
class NHInt_loss(nn.Module):
    def __init__(self, grid_layer):
        super().__init__()
        self.grid_layer: GridLayer = grid_layer
        self.eps = 1e-6

    def forward(self, output, target=None, sample_configs={}, mask=None,**kwargs):
        output_nh, _ = self.grid_layer.get_nh(output, **sample_configs)

        target_nh, _ = self.grid_layer.get_nh(target, **sample_configs)
    
        loss = (output_nh.abs().sum(dim=-2) - target_nh.abs().sum(dim=-2)).abs().mean()

        return loss
    
class NHVar_loss(nn.Module):
    def __init__(self, grid_layer):
        super().__init__()
        self.grid_layer: GridLayer = grid_layer
        self.eps = 1e-6

    def forward(self, output, target=None, sample_configs={}, mask=None,**kwargs):
        output_nh, _ = self.grid_layer.get_nh(output, **sample_configs)

        target_nh, _ = self.grid_layer.get_nh(target, **sample_configs)
        
        out_logstd = 0.5 * torch.log((output_nh).var(dim=-2) + self.eps)
        tgt_logstd = 0.5 * torch.log((target_nh).var(dim=-2) + self.eps)

        loss = (out_logstd - tgt_logstd).abs().mean()

        return loss
    
class GNLL_loss(nn.Module):
    def __init__(self, grid_layer=None):
        super().__init__()

        self.loss_fcn = torch.nn.GaussianNLLLoss() 

    def forward(self, output, target, **kwargs):
        output, output_var = output.chunk(2, dim=-1)
        loss = self.loss_fcn(output, target.view(*output.shape), output_var)
        return loss
    
class MSE_Hole_loss(nn.Module):
    def __init__(self, grid_layer=None):
        super().__init__()

        self.loss_fcn = torch.nn.MSELoss() 

    def forward(self, output, target, mask=None, **kwargs):
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
    def __init__(self, lambda_dict, grid_layer=None):
        super().__init__()

        self.loss_fcns = []
        for target, lambda_ in lambda_dict.items():
            if float(lambda_) > 0:
                self.loss_fcns.append({'lambda': float(lambda_), 
                                        'fcn': globals()[target](grid_layer=grid_layer)})
    
    def forward(self, output, target, mask=None, sample_configs={}, prefix=''):
        loss_dict = {}
        total_loss = 0

        for loss_fcn in self.loss_fcns:
            loss = loss_fcn['fcn'](output, target, mask=mask, sample_configs=sample_configs)
            loss_dict[prefix + loss_fcn['fcn']._get_name()] = loss.item()
            total_loss = total_loss + loss_fcn['lambda'] * loss
        
        return total_loss, loss_dict


class Grad_loss(nn.Module):
    def __init__(self, grid_layer):
        super().__init__()
        self.grid_layer: GridLayer = grid_layer
        self.loss_fcn = torch.nn.KLDivLoss(log_target=True, reduction='batchmean') 

    def forward(self, output, target, sample_configs={}, mask=None,**kwargs):
        indices_in  = sample_configs["indices_layers"][int(self.grid_layer.global_zoom)] if sample_configs is not None and isinstance(sample_configs, dict) else None
        output_nh, _ = self.grid_layer.get_nh(output, indices_in, sample_configs)
        
        target_nh, _ = self.grid_layer.get_nh(target.view(output.shape), indices_in, sample_configs)

        nh_diff_output = 1+(output_nh[:,:,[0]] - output_nh[:,:,1:])/output_nh[:,:,[0]]
        nh_diff_target = 1+(target_nh[:,:,[0]] - target_nh[:,:,1:])/target_nh[:,:,[0]]

        nh_diff_output = nh_diff_output.view(output_nh.shape[0],-1).clamp(min=0, max=1)
        nh_diff_target = nh_diff_target.view(output_nh.shape[0],-1).clamp(min=0, max=1)

        nh_diff_output = nn.functional.log_softmax(nh_diff_output, dim=-1)
        nh_diff_target = nn.functional.log_softmax(nh_diff_target, dim=-1)

        loss = self.loss_fcn(nh_diff_output, nh_diff_target)
        return loss

class NHTV_loss(nn.Module):
    def __init__(self, grid_layer):
        super().__init__()
        self.grid_layer: GridLayer = grid_layer

    def forward(self, output, target=None, sample_configs={}, mask=None,**kwargs):
        output_nh, _ = self.grid_layer.get_nh(output, **sample_configs)


        loss = (((output_nh[...,[0],:] - output_nh[...,1:,:]))**2).mean().sqrt()
        return loss
    

class NHTV_decay_loss(nn.Module):
    def __init__(self, grid_layer, tau=0.2):
        super().__init__()
        self.grid_layer: GridLayer = grid_layer
        self.tau = tau

    def forward(self, output, target=None, sample_configs={}, mask=None,**kwargs):
        output_nh, _ = self.grid_layer.get_nh(output, **sample_configs)
        target_nh, _ = self.grid_layer.get_nh(target, **sample_configs)

        nh_diff = (output_nh[...,[0],:] - output_nh[...,1:,:])**2
        target_diff = (target_nh[...,[0],:] - target_nh[...,1:,:]).abs()/(target_nh[...,[0],:].abs()+1e-6)

        loss = (torch.exp(-1*target_diff/self.tau)*nh_diff).mean().sqrt()

        return loss