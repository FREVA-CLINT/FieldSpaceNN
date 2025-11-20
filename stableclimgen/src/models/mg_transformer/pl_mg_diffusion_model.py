import torch
from pytorch_lightning.utilities import rank_zero_only

from .pl_mg_probabilistic import LightningProbabilisticModel
from ..diffusion.mg_gaussian_diffusion import GaussianDiffusion
from ..diffusion.mg_sampler import DDPMSampler, DDIMSampler
from ..mg_transformer.pl_mg_model import LightningMGModel, merge_sampling_dicts
from ...modules.grids.grid_utils import decode_zooms


class Lightning_MG_diffusion_transformer(LightningMGModel, LightningProbabilisticModel):
    def __init__(self,
                 model,
                 gaussian_diffusion: GaussianDiffusion,
                 lr_groups,
                 lambda_loss_dict,
                 weight_decay=0,
                 sampler="ddpm", n_samples=1, max_batchsize=-1, mg_encoder_config=None):
        super().__init__(
            model,
            lr_groups,
            lambda_loss_dict,
            weight_decay
        )

        self.gaussian_diffusion = gaussian_diffusion
        if sampler == "ddpm":
            self.sampler = DDPMSampler(self.gaussian_diffusion)
        else:
            self.sampler = DDIMSampler(self.gaussian_diffusion)
        self.n_samples = n_samples
        self.max_batchsize = max_batchsize


    def forward(self, x_zooms, sample_configs={}, mask_zooms=None, emb=None, out_zoom=None, pred_xstart=False):
        diffusion_steps, weights = self.gaussian_diffusion.get_diffusion_steps(x_zooms[max(x_zooms.keys())].shape[0],
                                                                               x_zooms[max(x_zooms.keys())].device)
        model_kwargs = {
            'sample_configs': sample_configs,
            'out_zoom': out_zoom
        }
        return self.gaussian_diffusion.training_losses(self.model, x_zooms, diffusion_steps, mask_zooms, emb, create_pred_xstart=pred_xstart, **model_kwargs)

    def training_step(self, batch, batch_idx):
        sample_configs = self.trainer.val_dataloaders.dataset.sampling_zooms_collate or self.trainer.val_dataloaders.dataset.sampling_zooms
        source, target, patch_index_zooms, mask, emb = batch

        sample_configs = merge_sampling_dicts(sample_configs, patch_index_zooms)

        loss, loss_dict, _, _, _ = self.get_losses(target.copy(), target, sample_configs,
                                                                         mask_zooms=mask, emb=emb, prefix='train',
                                                                         mode="diffusion")

        self.log_dict({"train/total_loss": loss.item()}, prog_bar=True)
        self.log_dict(loss_dict, logger=True)

        return loss


    def validation_step(self, batch, batch_idx):
        sample_configs = self.trainer.val_dataloaders.dataset.sampling_zooms_collate or self.trainer.val_dataloaders.dataset.sampling_zooms
        source, target, patch_index_zooms, mask, emb = batch

        max_zoom = max(target.keys())
        sample_configs = merge_sampling_dicts(sample_configs, patch_index_zooms)

        loss, loss_dict, output, output_comp, pred_xstart = self.get_losses(target.copy(), target, sample_configs,
                                                                            mask_zooms=mask, emb=emb, prefix='val',
                                                                            mode = "diffusion",
                                                                            pred_xstart=(batch_idx == 0 and rank_zero_only.rank==0))

        self.log_dict({"val/total_loss": loss.item()}, prog_bar=True)
        self.log_dict(loss_dict, logger=True)

        if batch_idx == 0 and rank_zero_only.rank==0:
            pred_xstart_comp = decode_zooms(pred_xstart.copy(), sample_configs=sample_configs, out_zoom=max_zoom)

            self.log_tensor_plot(source, pred_xstart, target, mask, sample_configs, emb, self.current_epoch, output_comp=pred_xstart_comp)

        return loss

    def predict_step(self, batch, batch_idx):
        # Call the desired parent's method directly
        # Note: Pass 'self' explicitly here
        return LightningProbabilisticModel.predict_step(self, batch, batch_idx)

    def _predict_step(self, source, target, patch_index_zooms, mask, emb):
        sample_configs = self.trainer.predict_dataloaders.dataset.sampling_zooms_collate or self.trainer.predict_dataloaders.dataset.sampling_zooms
        sample_configs = merge_sampling_dicts(sample_configs, patch_index_zooms)
        model_kwargs = {
            'sample_configs': sample_configs
        }
        max_zoom = max(target.keys())

        outputs = self.sampler.sample_loop(self.model, source, mask, progress=True, emb=emb, **model_kwargs)
        outputs = decode_zooms(outputs.copy(), sample_configs=sample_configs, out_zoom=max_zoom)
        return outputs