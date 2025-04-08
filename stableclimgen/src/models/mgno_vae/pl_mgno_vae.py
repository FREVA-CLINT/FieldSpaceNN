import os
import math
import torch.nn as nn

import lightning.pytorch as pl
import torch

from torch.optim import AdamW

from ..mgno_transformer.pl_probabilistic import LightningProbabilisticModel
from ...utils.visualization import scatter_plot
from ..mgno_transformer.pl_mgno_base_model import CosineWarmupScheduler, check_empty
from ..mgno_transformer.pl_mgno_base_model import LightningMGNOBaseModel


class MSE_loss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss_fcn = torch.nn.MSELoss() 

    def forward(self, output, target):
        loss = self.loss_fcn(output, target.view(output.shape))
        return loss


class Lightning_MGNO_VAE(LightningMGNOBaseModel, LightningProbabilisticModel):
    def __init__(self, model, lr_groups, kl_weight: float = 1e-6, n_samples=1, max_batchsize=-1, **base_model_args):
        super().__init__(model, lr_groups, **base_model_args)
        # maybe create multi_grid structure here?
        self.model = model

        self.lr_groups = lr_groups
        self.kl_weight = kl_weight

        self.n_samples = n_samples
        self.max_batchsize = max_batchsize

        self.save_hyperparameters(ignore=['model'])


    def training_step(self, batch, batch_idx):
        source, target, coords_input, coords_output, indices, mask, emb, dists_input = batch
        output, posterior, interp_x = self(source, coords_input=coords_input, coords_output=coords_output, indices_sample=indices, mask=mask, emb=emb, dists_input=dists_input)

        rec_loss, loss_dict = self.loss(output, target, mask=mask, indices_sample=indices, prefix='train/')

        # Compute KL divergence loss
        if self.kl_weight != 0.0:
            kl_loss = posterior.kl()
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
            loss = rec_loss + self.kl_weight * kl_loss
            loss_dict['train/kl_loss'] = self.kl_weight * kl_loss
        else:
            loss = rec_loss

        loss_dict['train/total_loss'] = loss.item()

        self.log_dict(loss_dict, prog_bar=True, sync_dist=False)
        return loss


    def validation_step(self, batch, batch_idx):
        source, target, coords_input, coords_output, indices, mask, emb, dists_input = batch
        coords_input, coords_output, indices, mask, emb, dists_input = check_empty(coords_input), check_empty(coords_output), check_empty(indices), check_empty(mask), check_empty(emb), check_empty(dists_input)
        output, posterior, interp_x = self(source, coords_input=coords_input, coords_output=coords_output, indices_sample=indices, mask=mask, emb=emb, dists_input=dists_input)

        rec_loss, loss_dict = self.loss(output, target, mask=mask, indices_sample=indices, prefix='val/')

        # Compute KL divergence loss
        if self.kl_weight != 0.0:
            kl_loss = posterior.kl()
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
            loss = rec_loss + self.kl_weight * kl_loss
            loss_dict['val/kl_loss'] = self.kl_weight * kl_loss
        else:
            loss = rec_loss

        loss_dict['val/total_loss'] = loss.item()

        # Calculate total loss

        if batch_idx == 0:
            if hasattr(self.model, "interpolator") and self.model.interpolate_input:
                input_inter = interp_x
                _, coords_input, _, _, _, _, dists_input = self.model.prepare_batch(source, coords_input=coords_input, input_dists=dists_input)
                _, density = self.model.interpolator(source,
                                                     mask=mask,
                                                     indices_sample=indices,
                                                     calc_density=True)
            else:
                input_inter = None
                density = None
            self.log_tensor_plot(source, output, target, coords_input, coords_output, mask, indices,f"tensor_plot_{self.current_epoch}", emb, input_inter=input_inter, input_density=density)

        self.log_dict(loss_dict, prog_bar=False, sync_dist=True)

        return loss

    def _predict_step(self, source, mask, emb, coords_input, coords_output, indices_sample):
        output, posterior = self(source, coords_input=coords_input, coords_output=coords_output, indices_sample=indices_sample, mask=mask, emb=emb)
        return output