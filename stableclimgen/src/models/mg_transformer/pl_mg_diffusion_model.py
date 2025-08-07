import os

import torch
from stableclimgen.src.models.mg_transformer.pl_mg_model import MGMultiLoss

from stableclimgen.src.modules.multi_grid.input_output import MG_Difference_Encoder, MG_Encoder
from pytorch_lightning.utilities import rank_zero_only

from .pl_mg_probabilistic import LightningProbabilisticModel
from ..diffusion.mg_gaussian_diffusion import GaussianDiffusion
from ..diffusion.mg_sampler import DDPMSampler, DDIMSampler
from ..mgno_transformer.pl_mgno_base_model import LightningMGNOBaseModel, check_empty
from ...modules.grids.grid_utils import decode_zooms


class Lightning_MG_diffusion_transformer(LightningMGNOBaseModel, LightningProbabilisticModel):
    def __init__(self,
                 model,
                 gaussian_diffusion: GaussianDiffusion,
                 lr_groups,
                 lambda_loss_dict,
                 weight_decay=0,
                 noise_std=0.0,
                 composed_loss = True,
                 sampler="ddpm", n_samples=1, max_batchsize=-1, mg_encoder_config=None):
        super().__init__(
            model,  # Main VAE model
            lr_groups,
            {},
            weight_decay=weight_decay,
            noise_std=noise_std
        )

        self.loss = MGMultiLoss(lambda_loss_dict, grid_layers=model.grid_layers, max_zoom=max(model.in_zooms))

        self.composed_loss = composed_loss

        # maybe create multi_grid structure here?
        self.model = model

        self.gaussian_diffusion = gaussian_diffusion
        if sampler == "ddpm":
            self.sampler = DDPMSampler(self.gaussian_diffusion)
        else:
            self.sampler = DDIMSampler(self.gaussian_diffusion)
        self.n_samples = n_samples
        self.max_batchsize = max_batchsize

        if mg_encoder_config:
            self.encoder = MG_Difference_Encoder(
                out_zooms=mg_encoder_config.out_zooms
            )
        else:
            self.encoder = None


    def forward(self, x, diffusion_steps, coords_input, coords_output, sample_configs={}, mask=None, emb=None, dists_input=None, create_pred_xstart=False):
        x, coords_input, coords_output, sample_configs, mask, emb, dists_input = self.prepare_inputs(x, coords_input, coords_output, sample_configs, mask, emb, dists_input)
        model_kwargs = {
            'coords_input': coords_input,
            'coords_output': coords_output,
            'sample_configs': sample_configs
        }
        targets, outputs, pred_xstart = self.gaussian_diffusion.training_losses(self.model, x, diffusion_steps, mask, emb, create_pred_xstart=create_pred_xstart, **model_kwargs)
        return targets, outputs, pred_xstart

    def training_step(self, batch, batch_idx):
        source, target, coords_input, coords_output, sample_configs, mask, emb, rel_dists_input, _ = batch
        max_zoom = max(target.keys())
        diffusion_steps, weights = self.gaussian_diffusion.get_diffusion_steps(target[max_zoom].shape[0], target[max_zoom].device)

        targets, outputs, _ = self(target, diffusion_steps, coords_input, coords_output, sample_configs, mask, emb, rel_dists_input)

        loss, loss_dict = self.loss(outputs, targets, mask=mask, sample_configs=sample_configs, prefix='train/')

        self.log_dict({"train/total_loss": loss.item()}, prog_bar=True)
        self.log_dict(loss_dict, logger=True)

        return loss


    def validation_step(self, batch, batch_idx):
        source, target, coords_input, coords_output, sample_configs, mask, emb, rel_dists_input, _ = batch
        coords_input, coords_output, mask, rel_dists_input = check_empty(coords_input), check_empty(coords_output), check_empty(mask), check_empty(rel_dists_input)
        max_zoom = max(target.keys())
        diffusion_steps, weights = self.gaussian_diffusion.get_diffusion_steps(target[max_zoom].shape[0], target[max_zoom].device)

        targets, outputs, pred_xstart = self(target.copy(), diffusion_steps, coords_input, coords_output, sample_configs, mask, emb, rel_dists_input, create_pred_xstart=(batch_idx == 0 and rank_zero_only))
        loss, loss_dict = self.loss(outputs, targets, mask=mask, sample_configs=sample_configs, prefix='val/')


        self.log_dict({"val/total_loss": loss.item()}, prog_bar=True)
        self.log_dict(loss_dict, logger=True)

        if batch_idx == 0 and rank_zero_only.rank==0:
            self.log_tensor_plot(source, pred_xstart, target, mask, sample_configs, emb, self.current_epoch)

        return loss

    def predict_step(self, batch, batch_idx):
        # Call the desired parent's method directly
        # Note: Pass 'self' explicitly here
        return LightningProbabilisticModel.predict_step(self, batch, batch_idx)

    def _predict_step(self, source, target, mask, emb, coords_input, coords_output, sample_configs, dists_input):
        interp_x, coords_input, coords_output, sample_configs, mask, emb, dists_input = self.prepare_inputs(target,
                                                                                                         coords_input,
                                                                                                         coords_output,
                                                                                                         sample_configs,
                                                                                                         mask, emb,
                                                                                                         dists_input)
        model_kwargs = {
            'coords_input': coords_input,
            'coords_output': coords_output,
            'sample_configs': sample_configs
        }

        outputs = self.sampler.sample_loop(self.model, source, mask, progress=True, emb=emb, **model_kwargs)
        outputs = decode_zooms(outputs, max(outputs.keys()))
        return outputs