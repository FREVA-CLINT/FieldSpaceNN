import os

import torch

from .pl_probabilistic import LightningProbabilisticModel
from ..diffusion.gaussian_diffusion import GaussianDiffusion
from ..diffusion.sampler import DDPMSampler, DDIMSampler
from ..mgno_transformer.pl_mgno_base_model import LightningMGNOBaseModel, check_empty


class Lightning_MGNO_diffusion_transformer(LightningMGNOBaseModel, LightningProbabilisticModel):
    def __init__(self, model, gaussian_diffusion: GaussianDiffusion, lr_groups, sampler="ddpm", n_samples=1, max_batchsize=-1, **base_model_args):
        super().__init__(model, lr_groups, **base_model_args)

        # maybe create multi_grid structure here?
        self.model = model

        self.lr_groups = lr_groups
        self.gaussian_diffusion = gaussian_diffusion
        if sampler == "ddpm":
            self.sampler = DDPMSampler(self.gaussian_diffusion)
        else:
            self.sampler = DDIMSampler(self.gaussian_diffusion)
        self.n_samples = n_samples
        self.max_batchsize = max_batchsize
        self.save_hyperparameters(ignore=['model'])


    def forward(self, x, diffusion_steps, coords_input, coords_output, sample_configs={}, mask=None, emb=None, dists_input=None):
        x, mask, emb, model_kwargs = self.prepare_inputs(x, coords_input, coords_output, sample_configs, mask, emb, dists_input)

        l_dict, output = self.gaussian_diffusion.training_losses(self.model, x, diffusion_steps.view(-1), mask, emb, **model_kwargs)
        return l_dict, output

    def prepare_inputs(self, x, coords_input, coords_output, sample_configs={}, mask=None, emb=None, dists_input=None):
        interp_x, coords_input, coords_output, sample_configs, mask, emb, dists_input = super().prepare_inputs(x, coords_input, coords_output, sample_configs, mask, emb, dists_input)
        model_kwargs = {
            'coords_input': coords_input,
            'coords_output': coords_output,
            'sample_configs': sample_configs
        }
        if self.interpolator:
            model_kwargs['condition'] = interp_x
        emb['CoordinateEmbedder'] = self.model.grid_layer_max.get_coordinates(**sample_configs)

        return x, mask, emb, model_kwargs


    def training_step(self, batch, batch_idx):
        source, target, coords_input, coords_output, sample_configs, mask, emb, rel_dists_input, _ = batch
        diffusion_steps, weights = self.gaussian_diffusion.get_diffusion_steps(target.shape[0], target.device)
        diffusion_steps = diffusion_steps.unsqueeze(-1).repeat(1, target.shape[1])

        l_dict, _ = self(target, diffusion_steps, coords_input, coords_output, sample_configs, mask, emb, rel_dists_input)

        loss = (l_dict["loss"] * weights).mean()
        self.log_dict({"train/total_loss": loss.item()}, prog_bar=True)

        return loss


    def validation_step(self, batch, batch_idx):
        source, target, coords_input, coords_output, sample_configs, mask, emb, rel_dists_input, _ = batch

        loss_dict = {}
        loss = []

        # Iterate over batch items and compute validation loss for each
        for i in range(target.shape[0]):
            diff_steps = self.gaussian_diffusion.diffusion_steps
            n_samples = 4
            t = torch.tensor([(diff_steps // n_samples) * x for x in range(n_samples - 1)] + [diff_steps-1]).to(target.device)
            t = t.unsqueeze(-1).repeat(1, target.shape[1])
            in_source = torch.stack(n_samples * [source[i]])
            in_target = torch.stack(n_samples * [target[i]])
            in_coords_input = torch.stack(n_samples * [coords_input[i]])
            in_dists_input = torch.stack(n_samples * [rel_dists_input[i]])
            in_coords_output = torch.stack(n_samples * [coords_output[i]])
            in_mask = torch.stack(n_samples * [mask[i]]).unsqueeze(-1)
            in_sample_configs = {k: torch.stack(n_samples*[v[i]]) for k, v in sample_configs.items()}
            in_emb = {k: torch.stack(n_samples*[v[i]]) for k, v in emb.items()}

            l_dict, output = self(in_target, t, in_coords_input, in_coords_output, in_sample_configs, in_mask, in_emb, in_dists_input)

            in_coords_input, in_coords_output, in_sample_configs, in_mask, in_emb, in_dists_input = check_empty(
                in_coords_input), check_empty(
                in_coords_output), check_empty(in_sample_configs), check_empty(in_mask), check_empty(in_emb), check_empty(
                in_dists_input)

            for k, v in l_dict.items():
                for ti in range(t.shape[0]):
                    loss_dict[f'val/step_{t[ti][0].item()}_{k}'] = v[ti]
                loss.append(v.mean())

            if batch_idx == 0 and i == 0:
                if hasattr(self, "interpolator") and self.interpolator is not None:
                    input_inter, input_density = self.interpolator(in_source, mask=in_mask, input_coords=in_coords_input,
                                                                   sample_configs=in_sample_configs, calc_density=True,
                                                                   input_dists=in_rel_dists_input)
                else:
                    input_inter = None
                    input_density = None
                has_var = hasattr(self.model, 'predict_var') and self.model.predict_var
                self.log_tensor_plot(in_source, output, in_target, in_coords_input, in_coords_output, in_mask, in_sample_configs,
                                     f"tensor_plot_{int(self.current_epoch)}", in_emb, input_inter=input_inter,
                                     input_density=input_density, has_var=has_var)

        self.log_dict(loss_dict, sync_dist=True)
        self.log('val_loss', torch.stack(loss).mean(), sync_dist=True)
        return self.log_dict

    def predict_step(self, batch, batch_idx):
        # Call the desired parent's method directly
        # Note: Pass 'self' explicitly here
        return LightningProbabilisticModel.predict_step(self, batch, batch_idx)

    def _predict_step(self, source, mask, emb, coords_input, coords_output, sample_configs, input_dists):
        source, mask, emb, model_kwargs = self.prepare_inputs(source, coords_input, coords_output, sample_configs, mask, emb, input_dists)

        return self.sampler.sample_loop(self.model, source, mask,
                                        progress=True, emb=emb, **model_kwargs)
