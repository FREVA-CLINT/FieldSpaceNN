import os
import math
import torch.nn as nn

import lightning.pytorch as pl
import torch
from typing import Dict
from torch.optim import AdamW

from pytorch_lightning.utilities import rank_zero_only
from ...utils.visualization import scatter_plot
from ...modules.grids.grid_layer import GridLayer, Interpolator
from ...models.mgno_transformer.pl_mgno_base_model import LightningMGNOBaseModel


def check_empty(x):

    if isinstance(x, torch.Tensor):
        if x.numel()==0:
            return None
        else:
            return x      
    else:
        return x
    
class MG_MSE_loss(nn.Module):
    def __init__(self, grid_layer=None):
        super().__init__()

        self.loss_fcn = torch.nn.MSELoss() 

    def forward(self, output:Dict, target, **kwargs):

        loss = self.loss_fcn(output, target.view(output.shape))
        return loss

class LightningMGModel(LightningMGNOBaseModel):
    def __init__(self, 
                 model, 
                 lr_groups, 
                 lambda_loss_dict: dict, 
                 weight_decay=0, 
                 noise_std=0.0, 
                 interpolator_settings=None):
        
        super().__init__(
            model,  # Main VAE model
            lr_groups,
            lambda_loss_dict,
            weight_decay=weight_decay,
            noise_std=noise_std,
            interpolator_settings=interpolator_settings
        )
