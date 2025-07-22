import os

import torch
from stableclimgen.src.models.mg_transformer.pl_mg_model import MGMultiLoss

from stableclimgen.src.modules.multi_grid.input_output import MG_Difference_Encoder, MG_Encoder
from pytorch_lightning.utilities import rank_zero_only

from .pl_mg_probabilistic import LightningProbabilisticModel
from ..diffusion.mg_gaussian_diffusion import GaussianDiffusion
from ..diffusion.mg_sampler import DDPMSampler, DDIMSampler
from ..mgno_transformer.pl_mgno_base_model import LightningMGNOBaseModel, check_empty


class Lightning_MG_diffusion_transformer(LightningMGNOBaseModel, LightningProbabilisticModel):
    def __init__(self,
                 model,
                 gaussian_diffusion: GaussianDiffusion,
                 lr_groups,
                 lambda_loss_dict,
                 weight_decay=0,
                 noise_std=0.0,
                 composed_loss = True,
                 interpolator_settings=None,
                 sampler="ddpm", n_samples=1, max_batchsize=-1, mg_encoder_config=None):
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


    def forward(self, x, diffusion_steps, coords_input, coords_output, sample_dict={}, mask=None, emb=None, dists_input=None, return_zooms=True, create_pred_xstart=False):
        interp_x, coords_input, coords_output, sample_dict, mask, emb, dists_input = self.prepare_inputs(x, coords_input, coords_output, sample_dict, mask, emb, dists_input)
        model_kwargs = {
            'coords_input': coords_input,
            'coords_output': coords_output,
            'sample_dict': sample_dict,
            'return_zooms': return_zooms
        }
        x = x.squeeze(-1)

        x_zooms = {int(sample_dict['zoom'][0]): x} if 'zoom' in sample_dict.keys() else {self.model.max_zoom: x}
        x_zooms = self.encoder(x_zooms, emb=emb, sample_dict=sample_dict)
        init_zoom = list(x_zooms.keys())[0]

        mask_zooms = {int(zoom): torch.zeros_like(x_zooms[zoom]).bool() if zoom == init_zoom else torch.ones_like(x_zooms[zoom]).bool() for zoom in x_zooms.keys()}

        targets, outputs, pred_xstart = self.gaussian_diffusion.training_losses(self.model, x_zooms, diffusion_steps, mask_zooms, emb, create_pred_xstart=create_pred_xstart, **model_kwargs)
        return targets, outputs, pred_xstart

    def training_step(self, batch, batch_idx):
        source, target, coords_input, coords_output, sample_dict, mask, emb, rel_dists_input, _ = batch
        diffusion_steps, weights = self.gaussian_diffusion.get_diffusion_steps(target.shape[0], target.device)

        targets, outputs, _ = self(target, diffusion_steps, coords_input, coords_output, sample_dict, mask, emb, rel_dists_input, return_zooms=(self.composed_loss==False))

        loss, loss_dict = self.loss(outputs, targets, mask=mask, sample_dict=sample_dict, prefix='train/')

        self.log_dict({"train/total_loss": loss.item()}, prog_bar=True)
        self.log_dict(loss_dict, logger=True)

        return loss


    def validation_step(self, batch, batch_idx):
        source, target, coords_input, coords_output, sample_dict, mask, emb, rel_dists_input, _ = batch
        diffusion_steps, weights = self.gaussian_diffusion.get_diffusion_steps(target.shape[0], target.device)

        targets, outputs, pred_xstart = self(target, diffusion_steps, coords_input, coords_output, sample_dict, mask, emb, rel_dists_input, return_zooms=(self.composed_loss==False), create_pred_xstart=(batch_idx == 0 and rank_zero_only))
        loss, loss_dict = self.loss(outputs, targets, mask=mask, sample_dict=sample_dict, prefix='val/')

        self.log_dict({"val/total_loss": loss.item()}, prog_bar=True)
        self.log_dict(loss_dict, logger=True)

        if batch_idx == 0 and rank_zero_only:
            pred_xstart = self.model.decoder(pred_xstart, emb=emb, sample_dict=sample_dict)
            pred_xstart = pred_xstart[self.model.max_zoom]

            coords_input, coords_output, mask, rel_dists_input = check_empty(coords_input), check_empty(
                coords_output), check_empty(mask), check_empty(rel_dists_input)
            sample_dict = self.prepare_sample_dict(sample_dict)

            if hasattr(self, "interpolator") and self.interpolator is not None:
                input_inter, input_density = self.interpolator(source, mask=mask, input_coords=coords_input,
                                                               sample_dict=sample_dict, calc_density=True,
                                                               input_dists=rel_dists_input)
            else:
                input_inter = None
                input_density = None
            has_var = hasattr(self.model, 'predict_var') and self.model.predict_var
            self.log_tensor_plot(source, pred_xstart, target, coords_input, coords_output, mask, sample_dict,
                                 f"tensor_plot_{int(self.current_epoch)}_diff{diffusion_steps[0].item()}", emb, input_inter=input_inter,
                                 input_density=input_density, has_var=has_var)

        return loss

    def predict_step(self, batch, batch_idx):
        # Call the desired parent's method directly
        # Note: Pass 'self' explicitly here
        return LightningProbabilisticModel.predict_step(self, batch, batch_idx)

    def _predict_step(self, source, target, mask, emb, coords_input, coords_output, sample_dict, dists_input):
        interp_x, coords_input, coords_output, sample_dict, mask, emb, dists_input = self.prepare_inputs(target,
                                                                                                         coords_input,
                                                                                                         coords_output,
                                                                                                         sample_dict,
                                                                                                         mask, emb,
                                                                                                         dists_input)
        model_kwargs = {
            'coords_input': coords_input,
            'coords_output': coords_output,
            'sample_dict': sample_dict,
            'return_zooms': (self.composed_loss==False)
        }
        target = target.squeeze(-1)

        x_zooms = {int(sample_dict['zoom'][0]): target} if 'zoom' in sample_dict.keys() else {self.model.max_zoom: target}
        x_zooms = self.encoder(x_zooms, emb=emb, sample_dict=sample_dict)
        init_zoom = list(x_zooms.keys())[0]

        mask_zooms = {int(zoom): torch.zeros_like(x_zooms[zoom]).bool() if zoom == init_zoom else torch.ones_like(
            x_zooms[zoom]).bool() for zoom in x_zooms.keys()}

        outputs = self.sampler.sample_loop(self.model, x_zooms, mask_zooms, progress=True, emb=emb, **model_kwargs)
        outputs = self.model.decoder(outputs, emb=emb, sample_dict=sample_dict)[self.model.max_zoom]
        return outputs