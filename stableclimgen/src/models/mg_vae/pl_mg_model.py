import os
import math
import torch.nn as nn

import lightning.pytorch as pl
import torch
from typing import Dict
from torch.optim import AdamW

from pytorch_lightning.utilities import rank_zero_only


from ..mg_transformer.pl_mg_probabilistic import LightningProbabilisticModel
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


class LightningMGVAEModel(LightningMGNOBaseModel, LightningProbabilisticModel):
    def __init__(self, 
                 model, 
                 lr_groups,
                 lambda_loss_dict: dict,
                 kl_weight: float = 1e-6,
                 weight_decay=0,
                 noise_std=0.0,
                 decomposed_loss = True,
                 n_samples=1, max_batchsize=-1):
        
        super().__init__(
            model,  # Main VAE model
            lr_groups,
            {},
            weight_decay=weight_decay,
            noise_std=noise_std
        )

        self.loss = MGMultiLoss(lambda_loss_dict, grid_layers=model.grid_layers, max_zoom=model.max_zoom)
        self.kl_weight = kl_weight
        self.n_samples = n_samples
        self.max_batchsize = max_batchsize

        self.decomposed_loss = decomposed_loss

    def forward(self, x, coords_input, coords_output, sample_dict={}, mask=None, emb=None, dists_input=None, return_zooms=True):
        x, coords_input, coords_output, sample_dict, mask, emb, dists_input = self.prepare_inputs(x, coords_input, coords_output, sample_dict, mask, emb, dists_input)
        x, posterior = self.model(x, coords_input=coords_input, coords_output=coords_output, sample_dict=sample_dict, mask=mask, emb=emb)
        return x, posterior
    
    def training_step(self, batch, batch_idx):
        source, target, coords_input, coords_output, sample_dict, mask, emb, rel_dists_input, _ = batch
        output, posterior_zooms = self(source, coords_input=coords_input, coords_output=coords_output, sample_dict=sample_dict, mask=mask, emb=emb, dists_input=rel_dists_input)
        
        if not self.decomposed_loss:
            max_zoom = max(target.keys())
            target = {max_zoom: decode_zooms(target,max_zoom)}
            output = {max_zoom: decode_zooms(output,max_zoom)}

        rec_loss, loss_dict = self.loss(output, target, mask=mask, sample_dict=sample_dict, prefix='train/')

        # Compute KL divergence loss
        if self.kl_weight != 0.0:
            kl_loss = torch.stack([posterior.kl() for posterior in posterior_zooms.values()]).mean(dim=0)
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
            loss = rec_loss + self.kl_weight * kl_loss
            loss_dict['train/kl_loss'] = self.kl_weight * kl_loss
        else:
            loss = rec_loss

        loss_dict['train/total_loss'] = loss.item()

        self.log_dict(loss_dict, prog_bar=True, sync_dist=False)

        return loss
    

    def validation_step(self, batch, batch_idx):

        source, target, coords_input, coords_output, sample_dict, mask, emb, rel_dists_input, _ = batch

        coords_input, coords_output, mask, rel_dists_input = check_empty(coords_input), check_empty(coords_output), check_empty(mask), check_empty(rel_dists_input)
        sample_dict = self.prepare_sample_dict(sample_dict)

        source_ = {k: v.clone() for k, v in source.items()}
        output, posterior_zooms = self(source, coords_input=coords_input, coords_output=coords_output, sample_dict=sample_dict, mask=mask, emb=emb, dists_input=rel_dists_input)

        target_loss = {k: v.clone() for k, v in target.items()}
        output_loss = {k: v.clone() for k, v in output.items()}
        
        max_zoom = max(target.keys())
        if not self.decomposed_loss:
            target_loss = {max_zoom: decode_zooms(target_loss,max_zoom)}
            output_loss = {max_zoom: decode_zooms(output_loss,max_zoom)}

        rec_loss, loss_dict = self.loss(output, target_loss, mask=mask, sample_dict=sample_dict, prefix='val/')

        # Compute KL divergence loss
        if self.kl_weight != 0.0:
            kl_loss = torch.stack([posterior.kl() for posterior in posterior_zooms.values()]).mean(dim=0)
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
            loss = rec_loss + self.kl_weight * kl_loss
            loss_dict['val/kl_loss'] = self.kl_weight * kl_loss
        else:
            loss = rec_loss

        loss_dict['val/total_loss'] = loss.item()

        self.log_dict(loss_dict, prog_bar=True, sync_dist=False)

        if batch_idx == 0 and rank_zero_only and (source_.device in ['cuda:0','cpu','mps']):
            has_var = hasattr(self.model,'predict_var') and self.model.predict_var

            source_p = decode_zooms(source_,max_zoom)
            output_p = decode_zooms(output,max_zoom)
            target_p = decode_zooms(target,max_zoom)
            mask = mask[max_zoom] if mask is not None else None
            self.log_tensor_plot(source_p, output_p, target_p, coords_input, coords_output, mask, sample_dict,f"tensor_plot_{int(self.current_epoch)}", emb, has_var=has_var)     

        return loss


    def predict_step(self, batch, batch_idx):
        # Call the desired parent's method directly
        # Note: Pass 'self' explicitly here
        return LightningProbabilisticModel.predict_step(self, batch, batch_idx)

    def _predict_step(self, source, mask, emb, coords_input, coords_output, sample_dict, dists_input):
        sample_dict = self.prepare_sample_dict(sample_dict)
        x, coords_input, coords_output, sample_dict, mask, emb, dists_input = self.prepare_inputs(source,
                                                                                                  coords_input,
                                                                                                  coords_output,
                                                                                                  sample_dict, mask,
                                                                                                  emb, dists_input)
        if self.mode == "encode_decode":
            output, posterior = self.model(source, coords_input=coords_input, coords_output=coords_output,
                                           sample_dict=sample_dict, mask=mask, emb=emb, return_zooms=False)
        elif self.mode == "encode":
            output = self.model.encode(x, coords_input=coords_input, coords_output=coords_output, sample_dict=sample_dict,
                                       mask=mask, emb=emb)
        elif self.mode == "decode":
            output = self.model.decode(x, coords_input=coords_input, coords_output=coords_output,
                                       sample_dict=sample_dict,
                                       mask=mask, emb=emb, return_zooms=False)
        output = output[list(output.keys())[0]]
        return output