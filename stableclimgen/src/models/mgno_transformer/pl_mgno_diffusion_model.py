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


    def forward(self, x, diffusion_steps, coords_input, coords_output, indices_sample=None, mask=None, emb=None, dists_input=None):
        b, nt, n = x.shape[:3]
        x, mask, emb, model_kwargs = self.prepare_inputs(x, coords_input, coords_output, indices_sample, mask, emb, dists_input)
        model_kwargs.pop("dists_input")
        l_dict, output = self.gaussian_diffusion.training_losses(self.model, x, diffusion_steps.view(-1), mask, emb, **model_kwargs)
        return l_dict, output.view(b, nt, *output.shape[1:])

    def prepare_inputs(self, x, coords_input, coords_output, indices_sample=None, mask=None, emb=None, dists_input=None):
        coords_input, coords_output, indices_sample, mask, emb, dists_input = check_empty(coords_input), check_empty(
            coords_output), check_empty(indices_sample), check_empty(mask), check_empty(emb), check_empty(dists_input)
        x, coords_input, coords_output, indices_sample, mask, emb, dists_input = self.prepare_batch(x, coords_input,
                                                                                                    coords_output,
                                                                                                    indices_sample,
                                                                                                    mask, emb,
                                                                                                    dists_input)
        indices_sample, coords_input, coords_output, dists_input = self.prepare_coords_indices(coords_input,
                                                                                               coords_output=coords_output,
                                                                                               indices_sample=indices_sample,
                                                                                               input_dists=dists_input)

        if self.interpolator:
            interp_x, density_map = self.interpolator(x,
                                                      mask=mask,
                                                      calc_density=True,
                                                      indices_sample=indices_sample,
                                                      input_coords=coords_input,
                                                      input_dists=dists_input)
            emb["DensityEmbedder"] = 1 - density_map.transpose(-2, -1)
            emb["UncertaintyEmbedder"] = (density_map.transpose(-2, -1), emb['VariableEmbedder'])
            mask = None
            condition = interp_x.unsqueeze(-3)
        else:
            condition = None
        model_kwargs = {
            'coords_input': coords_input,
            'coords_output': coords_output,
            'indices_sample': indices_sample,
            'dists_input': dists_input
        }
        if torch.is_tensor(condition):
            model_kwargs['condition'] = condition
        return x, mask, emb, model_kwargs


    def training_step(self, batch, batch_idx):
        source, target, coords_input, coords_output, indices, mask, emb, dists_input = batch
        diffusion_steps, weights = self.gaussian_diffusion.get_diffusion_steps(target.shape[0], target.device)
        diffusion_steps = diffusion_steps.unsqueeze(-1).repeat(1, target.shape[1])

        l_dict, _ = self(target, diffusion_steps, coords_input, coords_output, indices, mask.unsqueeze(-1), emb, dists_input)

        loss = (l_dict["loss"] * weights).mean()
        self.log_dict({"train/total_loss": loss.item()}, prog_bar=True)

        return loss


    def validation_step(self, batch, batch_idx):
        if batch_idx != 0:
            return
        source, target, coords_input, coords_output, indices, mask, emb, dists_input = batch

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
            in_dists_input = torch.stack(n_samples * [dists_input[i]])
            in_coords_output = torch.stack(n_samples * [coords_output[i]])
            in_mask = torch.stack(n_samples * [mask[i]]).unsqueeze(-1)
            if torch.is_tensor(indices):
                in_indices = torch.stack(n_samples*[indices[i]])
            else:
                in_indices = {"sample": torch.stack(n_samples*[indices["sample"][i]]), "sample_level": torch.stack(n_samples*[indices["sample_level"][i]])}
            if emb:
                in_emb = {"VariableEmbedder": torch.stack(n_samples * [emb["VariableEmbedder"][i]]),
                          "TimeEmbedder": torch.stack(n_samples * [emb["TimeEmbedder"][i]])}
            l_dict, output = self(in_target, t, in_coords_input, in_coords_output, in_indices, in_mask, in_emb, in_dists_input)

            in_coords_input, in_coords_output, in_indices, in_mask, in_emb, in_dists_input = check_empty(
                in_coords_input), check_empty(
                in_coords_output), check_empty(in_indices), check_empty(in_mask), check_empty(in_emb), check_empty(
                in_dists_input)

            for k, v in l_dict.items():
                for ti in range(t.shape[0]):
                    loss_dict[f'val/step_{t[ti][0].item()}_{k}'] = v[ti]
                loss.append(v.mean())

            if batch_idx == 0 and i == 0:
                b, nt, n, nv, nc = in_source.shape[:5]
                if self.interpolator:
                    input_inter, density = self.interpolator(in_source,
                                                             mask=in_mask,
                                                             indices_sample=in_indices,
                                                             calc_density=True,
                                                             input_dists=in_dists_input)
                    input_inter = input_inter.view(1, -1, n, nv, nc)
                    density = density.view(1, -1, n, nv, nc)
                else:
                    input_inter = None
                    density = None
                max_nt = 12

                self.log_tensor_plot(in_source.view(1, -1, n, nv, nc), output[:, :max_nt].view(1, -1, n, nv, nc), in_target.view(1, -1, n, nv, nc), in_coords_input, in_coords_output, in_mask.view(1, -1, n, nv, nc), in_indices, f"tensor_plot_{self.current_epoch}", in_emb, input_inter, density, n_samples * max_nt)

        self.log_dict(loss_dict, sync_dist=True)
        self.log('val_loss', torch.stack(loss).mean(), sync_dist=True)
        return self.log_dict

    def predict_step(self, batch, batch_idx):
        # Call the desired parent's method directly
        # Note: Pass 'self' explicitly here
        return LightningProbabilisticModel.predict_step(self, batch, batch_idx)

    def _predict_step(self, source, mask, emb, coords_input, coords_output, indices_sample, input_dists):
        source, mask, emb, model_kwargs = self.prepare_inputs(source, coords_input, coords_output, indices_sample, mask, emb, input_dists)
        return self.sampler.sample_loop(self.model, source, mask,
                                        progress=True, emb=emb, interpolator=self.interpolator, **model_kwargs)
