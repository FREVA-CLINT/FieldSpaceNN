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
from ...models.mgno_transformer.pl_mgno_base_model import LightningMGNOBaseModel,MSE_loss,NH_loss,GNLL_loss


def check_empty(x):

    if isinstance(x, torch.Tensor):
        if x.numel()==0:
            return None
        else:
            return x      
    else:
        return x

import torch.nn as nn

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

    def forward(self, output, target, mask=None, sample_dict=None, prefix=''):
        loss_dict = {}
        total_loss = 0

        for level_key in output:
            out = output[level_key]
            tgt = target[level_key]

            # Level-specific losses
            if level_key in self.level_loss_fcns:
                for loss_fcn in self.level_loss_fcns[level_key]:
                    loss = loss_fcn['fcn'](out, tgt, mask=mask, sample_dict=sample_dict)
                    name = f"{prefix}level{level_key}_{loss_fcn['fcn']._get_name()}"
                    loss_dict[name] = loss.item()
                    total_loss += loss_fcn['lambda'] * loss

            # Common losses
            for loss_fcn in self.common_loss_fcns:
                loss = loss_fcn['fcn'](out, tgt, mask=mask, sample_dict=sample_dict)
                name = f"{prefix}level{level_key}_{loss_fcn['fcn']._get_name()}"
                loss_dict[name] = loss.item()
                total_loss += loss_fcn['lambda'] * loss

        return total_loss, loss_dict

    
class MG_Diff_MSE_loss(nn.Module):
    def __init__(self, lambda_=1, grid_layer=None):
        super().__init__()

        self.loss_fcn = torch.nn.MSELoss()

    def forward(self, output:Dict, target:Dict, **kwargs):

        loss = self.loss_fcn(output, target.view(output.shape))
        return loss

class LightningMGVAEModel(LightningMGNOBaseModel):
    def __init__(self, 
                 model, 
                 lr_groups, 
                 lambda_loss_dict: dict, 
                 weight_decay=0, 
                 noise_std=0.0,
                 composed_loss = True,
                 interpolator_settings=None):
        
        super().__init__(
            model,  # Main VAE model
            lr_groups,
            {},
            weight_decay=weight_decay,
            noise_std=noise_std,
            interpolator_settings=interpolator_settings
        )

        self.loss = MGMultiLoss(lambda_loss_dict, grid_layers=model.grid_layers, max_zoom=model.max_zoom)

        self.composed_loss = composed_loss

    def forward(self, x, coords_input, coords_output, sample_dict={}, mask=None, emb=None, dists_input=None, return_zooms=True):
        x, coords_input, coords_output, sample_dict, mask, emb, dists_input = self.prepare_inputs(x, coords_input, coords_output, sample_dict, mask, emb, dists_input)
        x, posterior = self.model(x, coords_input=coords_input, coords_output=coords_output, sample_dict=sample_dict, mask=mask, emb=emb, return_zooms=return_zooms)
        return x, posterior
    
    def training_step(self, batch, batch_idx):
        source, target, coords_input, coords_output, sample_dict, mask, emb, rel_dists_input, _ = batch
        output, posterior = self(source, coords_input=coords_input, coords_output=coords_output, sample_dict=sample_dict, mask=mask, emb=emb, dists_input=rel_dists_input, return_zooms=(self.composed_loss==False))
        
        target = {self.model.max_zoom: target.squeeze(dim=-2)} if not isinstance(target, dict) else target

        if not self.composed_loss:
            target = self.model.encoder(target)

        rec_loss, loss_dict = self.loss(output, target, mask=mask, sample_dict=sample_dict, prefix='train/')

        # Compute KL divergence loss
        if self.kl_weight != 0.0:
            kl_loss = posterior.kl()
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
            loss = rec_loss + self.kl_weight * kl_loss
            loss_dict['train/kl_loss'] = self.kl_weight * kl_loss
        else:
            loss = rec_loss

        loss_dict['train/total_loss'] = loss.item()

        self.log_dict({"train/total_loss": loss.item()}, prog_bar=True)
        self.log_dict(loss_dict, logger=True)
        
        return loss
    

    def validation_step(self, batch, batch_idx):

        source, target, coords_input, coords_output, sample_dict, mask, emb, rel_dists_input, _ = batch

        coords_input, coords_output, mask, rel_dists_input = check_empty(coords_input), check_empty(coords_output), check_empty(mask), check_empty(rel_dists_input)
        sample_dict = self.prepare_sample_dict(sample_dict)

        output, posterior = self(source, coords_input=coords_input, coords_output=coords_output, sample_dict=sample_dict, mask=mask, emb=emb, dists_input=rel_dists_input, return_zooms=False)

        target_loss = {self.model.max_zoom: target.squeeze(dim=-2)} if not isinstance(target, dict) else target

        rec_loss, loss_dict = self.loss(output, target_loss, mask=mask, sample_dict=sample_dict, prefix='validate/')

        # Compute KL divergence loss
        if self.kl_weight != 0.0:
            kl_loss = posterior.kl()
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
            loss = rec_loss + self.kl_weight * kl_loss
            loss_dict['train/kl_loss'] = self.kl_weight * kl_loss
        else:
            loss = rec_loss

        loss_dict['val/total_loss'] = loss.item()

        self.log_dict({"val/total_loss": loss.item()}, prog_bar=True)
        self.log_dict(loss_dict, logger=True)


        output = output[list(output.keys())[0]]

        if batch_idx == 0 and rank_zero_only:
            if hasattr(self, "interpolator") and self.interpolator is not None:
                input_inter, input_density = self.interpolator(source, mask=mask, input_coords=coords_input, sample_dict=sample_dict, calc_density=True, input_dists=rel_dists_input)
            else:
                input_inter = None
                input_density = None
            has_var = hasattr(self.model,'predict_var') and self.model.predict_var
            self.log_tensor_plot(source, output, target, coords_input, coords_output, mask, sample_dict,f"tensor_plot_{int(self.current_epoch)}", emb, input_inter=input_inter, input_density=input_density, has_var=has_var)
            

        return loss


    def predict_step(self, batch, batch_idx):
        source, target, coords_input, coords_output, sample_dict, mask, emb, rel_dists_input, _ = batch
        output = self(source, coords_input=coords_input, coords_output=coords_output, sample_dict=sample_dict, mask=mask, emb=emb, dists_input=rel_dists_input, return_zooms=False)

        output = output[self.model.max_zoom]

        if hasattr(self.model,'predict_var') and self.model.predict_var:
            output, output_var = output.chunk(2, dim=-1)
        else:
            output_var = None

        output = {'output': output,
                'output_var': output_var,
                'mask': mask}
        return output