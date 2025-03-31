import os

import torch

from .pl_probabilistic import LightningProbabilisticModel
from ..diffusion.gaussian_diffusion import GaussianDiffusion
from ..diffusion.sampler import DDPMSampler, DDIMSampler
from ..mgno_transformer.pl_mgno_base_model import LightningMGNOBaseModel


class Lightning_MGNO_diffusion_transformer(LightningMGNOBaseModel, LightningProbabilisticModel):
    def __init__(self, model, gaussian_diffusion: GaussianDiffusion, lr_groups, sampler="ddpm", n_samples=1, max_batchsize=-1, **base_model_args):
        super().__init__(model, lr_groups, **base_model_args)
        LightningProbabilisticModel.__init__(self, n_samples=1, max_batchsize=-1)

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


    def forward(self, x, diffusion_steps, coords_input, coords_output, indices_sample=None, mask=None, emb=None):
        model_kwargs = {
            'coords_input': coords_input,
            'coords_output': coords_output,
            'indices_sample': indices_sample
        }
        return self.gaussian_diffusion.training_losses(self.model, x, diffusion_steps, mask, emb, **model_kwargs)

    def training_step(self, batch, batch_idx):
        source, target, coords_input, coords_output, indices, mask, emb = batch
        diffusion_steps, weights = self.gaussian_diffusion.get_diffusion_steps(target.shape[0], target.device)
        diffusion_steps = diffusion_steps.unsqueeze(-1).repeat(1, target.shape[1])

        l_dict, _ = self(target, diffusion_steps, coords_input, coords_output, indices, mask.unsqueeze(-1), emb)

        loss = (l_dict["loss"] * weights).mean()
        self.log_dict({"train/total_loss": loss.item()}, prog_bar=True)

        return loss


    def validation_step(self, batch, batch_idx):
        if batch_idx != 0:
            return
        source, target, coords_input, coords_output, indices, mask, emb = batch

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
            in_coords_output = torch.stack(n_samples * [coords_output[i]])
            in_mask = torch.stack(n_samples * [mask[i]]).unsqueeze(-1)
            if torch.is_tensor(indices):
                in_indices = torch.stack(n_samples*[indices[i]])
            else:
                in_indices = {"sample": torch.stack(n_samples*[indices["sample"][i]]), "sample_level": torch.stack(n_samples*[indices["sample_level"][i]])}
            if emb:
                in_emb = {"VariableEmbedder": torch.stack(n_samples * [emb["VariableEmbedder"][i]]),
                          "TimeEmbedder": torch.stack(n_samples * [emb["TimeEmbedder"][i]])}
            l_dict, output = self(in_target, t, in_coords_input, in_coords_output, in_indices, in_mask, in_emb)

            for k, v in l_dict.items():
                for ti in range(t.shape[0]):
                    loss_dict[f'val/step_{t[ti][0].item()}_{k}'] = v[ti]
                loss.append(v.mean())

            if batch_idx == 0 and i == 0:
                b, nt, n, nv, nc = in_source.shape[:5]
                self.log_tensor_plot(in_source.view(1, -1, n, nv, nc), output.view(1, -1, n, nv, nc), in_target.view(1, -1, n, nv, nc), in_coords_input, in_coords_output, in_mask.view(1, -1, n, nv, nc), in_indices, f"tensor_plot_{self.current_epoch}", in_emb, None, n_samples * 8)

        self.log_dict(loss_dict, sync_dist=True)
        self.log('val_loss', torch.stack(loss).mean(), sync_dist=True)
        return self.log_dict

    def _predict_step(self, source, mask, emb, coords_input, coords_output, indices_sample):
        return self.sampler.sample_loop(self.model, source, mask,
                                        progress=True, emb=emb, coords_input=coords_input,
                                        coords_output=coords_output, indices_sample=indices_sample)
