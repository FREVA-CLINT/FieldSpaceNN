import os
import math
import torch.nn as nn

import lightning.pytorch as pl
import torch
from pytorch_lightning.utilities import rank_zero_only

from torch.optim import AdamW

from ..mgno_transformer.pl_probabilistic import LightningProbabilisticModel
from ...modules.icon_grids.grid_layer import Interpolator
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
    def __init__(self, model, lr_groups, kl_weight: float = 1e-6, latent_loss_weight=None, n_samples=1, max_batchsize=-1, residual_interpolator_settings=None, **base_model_args):
        super().__init__(model, lr_groups, **base_model_args)
        # maybe create multi_grid structure here?
        self.model = model

        self.lr_groups = lr_groups
        self.kl_weight = kl_weight

        self.n_samples = n_samples
        self.max_batchsize = max_batchsize

        if latent_loss_weight:
            self.latent_loss = nn.MSELoss()
        else:
            self.latent_loss = None

        self.latent_loss_weight = latent_loss_weight

        if residual_interpolator_settings:
            self.residual_interpolator_down = Interpolator(self.model.grid_layers,
                             residual_interpolator_settings.get("search_level", 3),
                             0,
                             residual_interpolator_settings.get("bottleneck_level", 3),
                             residual_interpolator_settings.get("precompute", True),
                             residual_interpolator_settings.get("nh_inter", 3),
                             residual_interpolator_settings.get("power", 1)
                             )
            self.residual_interpolator_up = Interpolator(self.model.grid_layers,
                             residual_interpolator_settings.get("search_level", 3),
                             residual_interpolator_settings.get("bottleneck_level", 3),
                             0,
                             residual_interpolator_settings.get("precompute", True),
                             residual_interpolator_settings.get("nh_inter", 3),
                             residual_interpolator_settings.get("power", 1)
                             )
        else:
            self.residual_interpolator_up = self.residual_interpolator_down = None

        self.save_hyperparameters(ignore=['model'])

    def forward(self, x, coords_input, coords_output, indices_sample=None, mask=None, emb=None, dists_input=None):
        b, nt, n = x.shape[:3]
        coords_input, coords_output, indices_sample, mask, emb, dists_input = check_empty(coords_input), check_empty(coords_output), check_empty(indices_sample), check_empty(mask), check_empty(emb), check_empty(dists_input)
        x, coords_input, coords_output, indices_sample, mask, emb, dists_input = self.prepare_batch(x, coords_input, coords_output, indices_sample, mask, emb, dists_input)
        indices_sample, coords_input, coords_output, dists_input = self.prepare_coords_indices(coords_input,
                                                                                               coords_output=coords_output,
                                                                                               indices_sample=indices_sample,
                                                                                               input_dists=dists_input)
        if self.interpolator:
            x, density_map = self.interpolator(x,
                                              mask=mask,
                                              calc_density=True,
                                              indices_sample=indices_sample,
                                              input_coords=coords_input,
                                              input_dists=dists_input)
            emb["DensityEmbedder"] = 1 - density_map.transpose(-2, -1)
            emb["UncertaintyEmbedder"] = (density_map.transpose(-2, -1), emb['VariableEmbedder'])

            x = x.unsqueeze(-3)
            mask = None

        interp_x = None
        if self.residual_interpolator_up and self.residual_interpolator_down:
            interp_x, _ = self.residual_interpolator_down(x,
                                                          calc_density=False,
                                                          indices_sample=indices_sample)
            interp_x, _ = self.residual_interpolator_up(interp_x.unsqueeze(dim=-3),
                                                        calc_density=False,
                                                        indices_sample=indices_sample)

        x, posterior = self.model(x, coords_input=coords_input, coords_output=coords_output, indices_sample=indices_sample, mask=mask, emb=emb, residual=interp_x)
        return x.view(b, nt, *x.shape[1:]), posterior, interp_x.view(b, nt, *interp_x.shape[1:]) if torch.is_tensor(interp_x) else None

    def training_step(self, batch, batch_idx):
        source, target, coords_input, coords_output, indices, mask, emb, dists_input = batch
        output, posterior, interp_x = self(source, coords_input=coords_input, coords_output=coords_output, indices_sample=indices, mask=mask, emb=emb.copy(), dists_input=dists_input)

        loss, loss_dict = self.loss(output, target, mask=mask, indices_sample=indices, prefix='train/')

        if self.latent_loss:
            with torch.no_grad():
                _, posterior_unmasked, _ = self(target, coords_input=coords_input, coords_output=coords_output,
                                                indices_sample=indices, mask=torch.zeros_like(mask).bool(), emb=emb.copy(),
                                                dists_input=dists_input)
            latent_loss = self.latent_loss(posterior_unmasked.mean, posterior.mean)
            loss = loss + self.latent_loss_weight * latent_loss
            loss_dict['train/latent_loss'] = self.latent_loss_weight * latent_loss

        # Compute KL divergence loss
        if self.kl_weight != 0.0:
            kl_loss = posterior.kl()
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
            loss = loss + self.kl_weight * kl_loss
            loss_dict['train/kl_loss'] = self.kl_weight * kl_loss

        loss_dict['train/total_loss'] = loss.item()

        self.log_dict(loss_dict, prog_bar=True, sync_dist=False)
        return loss


    def validation_step(self, batch, batch_idx):
        source, target, coords_input, coords_output, indices, mask, emb, dists_input = batch
        coords_input, coords_output, indices, mask, emb, dists_input = check_empty(coords_input), check_empty(coords_output), check_empty(indices), check_empty(mask), check_empty(emb), check_empty(dists_input)
        output, posterior, interp_x = self(source, coords_input=coords_input, coords_output=coords_output, indices_sample=indices, mask=mask, emb=emb.copy(), dists_input=dists_input)

        loss, loss_dict = self.loss(output, target, mask=mask, indices_sample=indices, prefix='val/')

        if self.latent_loss:
            _, posterior_unmasked, _ = self(target, coords_input=coords_input, coords_output=coords_output,
                                            indices_sample=indices, mask=torch.zeros_like(mask).bool(), emb=emb.copy(),
                                            dists_input=dists_input)
            latent_loss = self.latent_loss(posterior_unmasked.mean, posterior.mean)
            loss = loss + self.latent_loss_weight * latent_loss
            loss_dict['val/latent_loss'] = self.latent_loss_weight * latent_loss

        # Compute KL divergence loss
        if self.kl_weight != 0.0:
            kl_loss = posterior.kl()
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
            loss = loss + self.kl_weight * kl_loss
            loss_dict['val/kl_loss'] = self.kl_weight * kl_loss

        loss_dict['val/total_loss'] = loss.item()

        # Calculate total loss

        if batch_idx == 0 and rank_zero_only:
            if self.interpolator:
                _, coords_input, _, _, _, _, dists_input = self.prepare_batch(source, coords_input=coords_input, input_dists=dists_input)
                _, density = self.interpolator(source,
                                               mask=mask,
                                               indices_sample=indices,
                                               calc_density=True,
                                               input_dists=dists_input)
            else:
                density = None
            self.log_tensor_plot(source, output, target, coords_input, coords_output, mask, indices,f"tensor_plot_{self.current_epoch}", emb, input_inter=interp_x, input_density=density)

        self.log_dict(loss_dict, prog_bar=False, sync_dist=True)

        return loss

    def predict_step(self, batch, batch_idx):
        # Call the desired parent's method directly
        # Note: Pass 'self' explicitly here
        return LightningProbabilisticModel.predict_step(self, batch, batch_idx)

    def _predict_step(self, source, mask, emb, coords_input, coords_output, indices_sample, input_dists):
        output, posterior, _ = self(source, coords_input=coords_input, coords_output=coords_output, indices_sample=indices_sample, mask=mask, emb=emb, dists_input=input_dists)
        return output