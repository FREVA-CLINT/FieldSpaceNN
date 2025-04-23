import os
import math
import torch.nn as nn
import copy

import lightning.pytorch as pl
import torch

from torch.optim import AdamW

from pytorch_lightning.utilities import rank_zero_only
from ...utils.visualization import scatter_plot
from ...modules.icon_grids.grid_layer import GridLayer, Interpolator
from .pl_mgno_base_model import LightningMGNOBaseModel

   
def get_update_mask(grid_layer_0, indices, mask, min_nh=1):

    mask_nh, _ = grid_layer_0.get_nh(mask[0], indices['indices_layers'][0] if indices is not None and isinstance(indices, dict) else None, indices)

    update_mask = (mask_nh == False).sum(dim=2,keepdim=False) >= min_nh

    return update_mask


def update_mask(mask, update_mask, inplace=False):
    
    if not inplace:
        mask_updated = mask.clone()
    else:
        mask_updated = mask

    mask_updated[update_mask.unsqueeze(dim=0)] = False

    return mask_updated
    

class LightningIterMGNOBaseModel(LightningMGNOBaseModel):
    def __init__(self, model, lr_groups, lambda_loss_dict: dict, weight_decay=0, noise_std=0.0, interpolator_settings=None, nh_step=1, finish_perc=1.,output_frequency=2):
        super().__init__(model, lr_groups, lambda_loss_dict, weight_decay=weight_decay, noise_std=noise_std, interpolator_settings=interpolator_settings)
        # maybe create multi_grid structure here?
        self.nh_step = nh_step
        self.finish_perc = finish_perc
        self.output_frequency = output_frequency


    def predict_step(self, batch, batch_idx):
        source, _, coords_input, coords_output, indices, mask, emb, dists_input = batch

        perc_iter_done = 1 - mask.sum()/mask.numel()

        iteration = 0

        outputs = []
        masks = []
        outputs_vars = []

        while perc_iter_done < (self.finish_perc-1e-5):
            
            emb_run = copy.deepcopy(emb)
            output = self(source, coords_input=coords_input, coords_output=coords_output, indices_sample=indices, mask=mask, emb=emb_run, dists_input=dists_input)

            if iteration % self.output_frequency == 0:
                outputs.append(output[...,0])
                outputs_vars.append(output[...,1] if hasattr(self.model,'predict_var') and self.model.predict_var else None)
                masks.append(mask)

            if iteration != 0:
                output = output.view(source.shape)
                output[mask_update.view(output.shape)==False] = source[mask_update.view(output.shape)==False]

            source = output.view(source.shape)

            mask_update = get_update_mask(self.model.grid_layers["0"], indices, mask, min_nh=self.nh_step)

            mask = update_mask(mask, mask_update, inplace=False)

            perc_iter_done = 1 - mask.sum()/mask.numel()

            print(perc_iter_done)
            iteration +=1
        
        outputs.append(output[...,0])
        outputs_vars.append(output[...,1] if hasattr(self.model,'predict_var') and self.model.predict_var else None)
        masks.append(mask)

        output = {'output': torch.stack(outputs, dim=1),
                'output_var': torch.stack(outputs_vars, dim=1) if outputs_vars[0] is not None else None,
                'mask': torch.stack(masks, dim=1)}
        
        return output