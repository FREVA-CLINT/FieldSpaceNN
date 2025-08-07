import os
import math
import torch.nn as nn

import lightning.pytorch as pl
import torch
from typing import Dict
from torch.optim import AdamW

from pytorch_lightning.utilities import rank_zero_only
from ...modules.grids.grid_utils import decode_zooms
from ...models.mgno_transformer.pl_mgno_base_model import LightningMGNOBaseModel,MSE_loss,GNLL_loss,NH_loss


def check_empty(x):

    if isinstance(x, torch.Tensor):
        if x.numel()==0:
            return None
        else:
            return x      
    else:
        return x

import torch.nn as nn

def merge_sampling_dicts(sample_configs, patch_index_zooms):
    for key, value in patch_index_zooms.items():
        if key in sample_configs:
            sample_configs[key]['patch_index'] = value

    return sample_configs
class MGMultiLoss(nn.Module):
    def __init__(self, lambda_dict, grid_layers=None, max_zoom=None):
        super().__init__()
        
        self.level_loss_fcns = {}   # {level: [{'lambda': x, 'fcn': ...}, ...]}
        self.common_loss_fcns = []  # [{'lambda': x, 'fcn': ...}, ...]

        for key, value in lambda_dict.items():
            if isinstance(key, (int, float)):  # Level-specific
                self.level_loss_fcns[key] = []
                for loss_name, lambda_ in value.items():
                    if float(lambda_) > 0:
                        self.level_loss_fcns[key].append({
                            'lambda': float(lambda_),
                            'fcn': globals()[loss_name](grid_layers[str(key)])
                        })
            else:  # Common loss
                if float(value) > 0:
                    self.common_loss_fcns.append({
                        'lambda': float(value),
                        'fcn': globals()[key](grid_layers[str(max_zoom)])
                    })

    def forward(self, output, target, mask=None, sample_configs={}, prefix=''):
        loss_dict = {}
        total_loss = 0

        for level_key in output:
            out = output[level_key]
            tgt = target[level_key]

            # Level-specific losses
            if level_key in self.level_loss_fcns:
                for loss_fcn in self.level_loss_fcns[level_key]:
                    loss = loss_fcn['fcn'](out, tgt, mask=mask, sample_configs=sample_configs)
                    name = f"{prefix}level{level_key}_{loss_fcn['fcn']._get_name()}"
                    loss_dict[name] = loss.item()
                    total_loss += loss_fcn['lambda'] * loss

            # Common losses
            for loss_fcn in self.common_loss_fcns:
                loss = loss_fcn['fcn'](out, tgt, mask=mask, sample_configs=sample_configs)
                name = f"{prefix}level{level_key}_{loss_fcn['fcn']._get_name()}"
                loss_dict[name] = loss.item()
                total_loss += loss_fcn['lambda'] * loss

        return total_loss, loss_dict

class LightningMGModel(LightningMGNOBaseModel):
    def __init__(self, 
                 model, 
                 lr_groups, 
                 lambda_loss_dict: dict, 
                 weight_decay=0, 
                 noise_std=0.0,
                 decomposed_loss=True):
        
        super().__init__(
            model,  # Main VAE model
            lr_groups,
            {},
            weight_decay=weight_decay,
            noise_std=noise_std
        )

        self.loss = MGMultiLoss(lambda_loss_dict, grid_layers=model.grid_layers, max_zoom=max(model.in_zooms))

        self.decomposed_loss = decomposed_loss

    def forward(self, x, sample_configs={}, mask_zooms=None, emb=None):
        x: torch.tensor = self.model(x, sample_configs=sample_configs, mask_zooms=mask_zooms, emb=emb)
        return x
    
    def training_step(self, batch, batch_idx):
        sample_configs = self.trainer.train_dataloader.dataset.sampling_zooms
        source, target, patch_index_zooms, mask, emb = batch

        sample_configs = merge_sampling_dicts(sample_configs, patch_index_zooms)
        output = self(source, sample_configs=sample_configs, mask_zooms=mask, emb=emb)

        if not self.decomposed_loss:
            max_zoom = max(target.keys())
            target = {max_zoom: decode_zooms(target,max_zoom)}
            output = {max_zoom: decode_zooms(output,max_zoom)}

        loss, loss_dict = self.loss(output, target, mask=mask, sample_configs=sample_configs, prefix='train/')

        self.log_dict({"train/total_loss": loss.item()}, prog_bar=True)
        self.log_dict(loss_dict, logger=True)
        
        return loss
    

    def validation_step(self, batch, batch_idx):
        
        sample_configs = self.trainer.val_dataloaders.dataset.sampling_zooms
        source, target, patch_index_zooms, mask, emb = batch

        sample_configs = merge_sampling_dicts(sample_configs, patch_index_zooms)
        output = self(source.copy(), sample_configs=sample_configs, mask_zooms=mask, emb=emb)

        target_loss = target.copy()
        output_loss = output.copy()
        
        max_zoom = max(target.keys())
        if not self.decomposed_loss:
            target_loss = {max_zoom: decode_zooms(target_loss, max_zoom)}
            output_loss = {max_zoom: decode_zooms(output_loss, max_zoom)}

        loss, loss_dict = self.loss(output_loss, target_loss, mask=mask, sample_configs=sample_configs, prefix='validate/')

        self.log_dict({"validate/total_loss": loss.item()}, prog_bar=True)
        self.log_dict(loss_dict, logger=True)

        if batch_idx == 0 and rank_zero_only.rank==0:
            self.log_tensor_plot(source, output, target,mask, sample_configs, emb, self.current_epoch)
            
        return loss


    def predict_step(self, batch, batch_idx):
        sample_configs = self.trainer.val_dataloaders.dataset.sampling_zooms
        source, target, patch_index_zooms, mask, emb = batch

        sample_configs = merge_sampling_dicts(sample_configs, patch_index_zooms)
        output = self(source.copy(), sample_configs=sample_configs, mask=mask, emb=emb)

        output = decode_zooms(output, max(output.keys()))
        mask = decode_zooms(mask, max(mask.keys()))

        if hasattr(self.model,'predict_var') and self.model.predict_var:
            output, output_var = output.chunk(2, dim=-1)
        else:
            output_var = None

        output = {'output': output,
                'output_var': output_var,
                'mask': mask}
        return output
    