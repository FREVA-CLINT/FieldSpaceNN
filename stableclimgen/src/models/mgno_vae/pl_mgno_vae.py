import os
import math
import torch.nn as nn

import lightning.pytorch as pl
import torch

from torch.optim import AdamW

from ...utils.visualization import scatter_plot
from ..mgno_transformer.pl_mgno_Transformer import CosineWarmupScheduler
from ..mgno_transformer.pl_mgno_Transformer import LightningMGNOTransformer


class MSE_loss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss_fcn = torch.nn.MSELoss() 

    def forward(self, output, target):
        loss = self.loss_fcn(output, target.view(output.shape))
        return loss


class Lightning_MGNO_VAE(LightningMGNOTransformer):
    def __init__(self, model, lr_groups, kl_weight: float = 1e-6):
        super().__init__(model, lr_groups)
        # maybe create multi_grid structure here?
        self.model = model

        self.lr_groups = lr_groups
        self.kl_weight = kl_weight
        self.save_hyperparameters(ignore=['model'])
        self.loss = MSE_loss()

    def forward(self, x, coords_input, coords_output, indices_sample=None, mask=None, emb=None):
        return self.model(x, coords_input=coords_input, coords_output=coords_output, indices_sample=indices_sample, mask=mask, emb=emb)

    def training_step(self, batch, batch_idx):
        source, target, coords_input, coords_output, indices, mask, emb = batch
        output, posterior = self(source, coords_input=coords_input, coords_output=coords_output, indices_sample=indices, mask=mask, emb=emb)
        rec_loss = self.loss(output, target)

        loss_dict = {}

        # Compute KL divergence loss
        if self.kl_weight != 0.0:
            kl_loss = posterior.kl()
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
            loss = rec_loss + self.kl_weight * kl_loss
            loss_dict['train/kl_loss'] = self.kl_weight * kl_loss
        else:
            loss = rec_loss

        loss_dict['train/rec_loss'] = rec_loss.mean()
        loss_dict['train/total_loss'] = loss.mean()

        self.log_dict(loss_dict, prog_bar=True, sync_dist=True)
        return loss.mean()


    def validation_step(self, batch, batch_idx):
        source, target, coords_input, coords_output, indices, mask, emb = batch
        output, posterior = self(source, coords_input=coords_input, coords_output=coords_output, indices_sample=indices, mask=mask, emb=emb)
        rec_loss = self.loss(output, target)

        loss_dict = {}

        # Compute KL divergence loss
        if self.kl_weight != 0.0:
            kl_loss = posterior.kl()
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
            loss = rec_loss + self.kl_weight * kl_loss
            loss_dict['val/kl_loss'] = self.kl_weight * kl_loss
        else:
            loss = rec_loss

        loss_dict['val/rec_loss'] = rec_loss.mean()
        loss_dict['val/total_loss'] = loss.mean()



        # Calculate total loss

        if batch_idx == 0:
            try:
                self.log_tensor_plot(source, output, target, coords_input, coords_output, mask, indices,f"tensor_plot_{self.current_epoch}", emb)
            except Exception as e:
                pass

        # Log validation losses
        self.log("val_loss", loss.mean(), sync_dist=True)
        self.log_dict(loss_dict, prog_bar=True, sync_dist=True)

        return loss
