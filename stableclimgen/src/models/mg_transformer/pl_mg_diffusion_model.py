import torch
from pytorch_lightning.utilities import rank_zero_only

from .pl_mg_probabilistic import LightningProbabilisticModel
from ...modules.diffusion.mg_gaussian_diffusion import MGGaussianDiffusion
from ...modules.diffusion.mg_sampler import DDPMSampler, DDIMSampler
from ..mg_transformer.pl_mg_model import LightningMGModel, merge_sampling_dicts
from ...modules.grids.grid_utils import decode_zooms


class Lightning_MG_diffusion_transformer(LightningMGModel, LightningProbabilisticModel):
    def __init__(self,
                 model,
                 gaussian_diffusion: MGGaussianDiffusion,
                 lr_groups,
                 lambda_loss_dict,
                 weight_decay=0,
                 sampler="ddpm", n_samples=1, max_batchsize=-1, mg_encoder_config=None, decode_zooms=True):
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
        self.decode_zooms = decode_zooms


    def forward(self, x_zooms_groups, sample_configs={}, mask_zooms_groups=None, emb_groups=None, out_zoom=None, pred_xstart=False, **kwargs):
        # Determine batch size from the first valid group
        first_valid_group = next((g for g in x_zooms_groups if g), None)
        if not first_valid_group:
            return [(None, None, None)] * len(x_zooms_groups)
        
        batch_size = first_valid_group[max(first_valid_group.keys())].shape[0]
        device = first_valid_group[max(first_valid_group.keys())].device
        
        diffusion_steps, _ = self.gaussian_diffusion.get_diffusion_steps(batch_size, device)
        
        model_kwargs = {
            'sample_configs': sample_configs,
            'out_zoom': out_zoom,
            **kwargs
        }
        
        return self.gaussian_diffusion.training_losses(
            self.model, x_zooms_groups, diffusion_steps, mask_zooms_groups, emb_groups, create_pred_xstart=pred_xstart, **model_kwargs
        )

    def training_step(self, batch, batch_idx):
        sample_configs = self.trainer.val_dataloaders.dataset.sampling_zooms_collate or self.trainer.val_dataloaders.dataset.sampling_zooms
        source_groups, target_groups, mask_groups, emb_groups, patch_index_zooms = batch

        sample_configs = merge_sampling_dicts(sample_configs, patch_index_zooms)

        loss, loss_dict, _, _ = self.get_losses(
            target_groups.copy(),
            target_groups,
            sample_configs,
            mask_groups=mask_groups,
            emb_groups=emb_groups,
            prefix='train',
            mode="diffusion",
        )

        self.log_dict({"train/total_loss": loss.item()}, prog_bar=True)
        self.log_dict(loss_dict, logger=True)

        return loss


    def validation_step(self, batch, batch_idx):
        sample_configs = self.trainer.val_dataloaders.dataset.sampling_zooms_collate or self.trainer.val_dataloaders.dataset.sampling_zooms
        source_groups, target_groups, mask_groups, emb_groups, patch_index_zooms = batch

        max_zooms = [max(target.keys()) for target in target_groups if target]
        max_zoom = max(max_zooms) if max_zooms else max(self.model.in_zooms)

        sample_configs = merge_sampling_dicts(sample_configs, patch_index_zooms)

        loss, loss_dict, _, pred_xstart = self.get_losses(
            [g.copy() for g in target_groups],
            target_groups,
            sample_configs,
            mask_groups=mask_groups,
            emb_groups=emb_groups,
            prefix='val',
            mode="diffusion",
            pred_xstart=(batch_idx == 0 and rank_zero_only.rank==0),
        )

        self.log_dict({"val/total_loss": loss.item()}, prog_bar=True)
        self.log_dict(loss_dict, logger=True)

        if batch_idx == 0 and rank_zero_only.rank==0:
            # Select the first group for visualization
            source_p_group = source_groups[0]
            target_p_group = target_groups[0]
            mask_p_group = mask_groups[0] if mask_groups and mask_groups[0] else None
            emb_p_group = emb_groups[0] if emb_groups and emb_groups[0] else None

            device = source_p_group[max(source_p_group.keys())].device
            ts = torch.tensor([(self.gaussian_diffusion.diffusion_steps // 4) * (x + 1) - 1 for x in range(4)]).to(device)

            for t in ts:
                # Create a single-item batch for visualization
                source_p = {zoom: source_p_group[zoom][0:1] for zoom in source_p_group.keys()}
                target_p = {zoom: target_p_group[zoom][0:1] for zoom in target_p_group.keys()}
                mask_p = {zoom: mask_p_group[zoom][0:1] for zoom in mask_p_group.keys()} if mask_p_group else None
                emb_p = {'VariableEmbedder': emb_p_group['VariableEmbedder'][0:1],
                         'TimeEmbedder': {int(zoom): emb_p_group['TimeEmbedder'][zoom][0:1] for zoom in emb_p_group['TimeEmbedder'].keys()}}
                patch_index_zooms_p = {zoom: patch_index_zooms[zoom][0:1] for zoom in patch_index_zooms.keys()}
                sample_configs_p = merge_sampling_dicts(self.trainer.val_dataloaders.dataset.sampling_zooms_collate or self.trainer.val_dataloaders.dataset.sampling_zooms, patch_index_zooms_p)
                model_kwargs = {'sample_configs': sample_configs_p}

                pred_xstart_outputs = self.gaussian_diffusion.training_losses(self.model, [target_p.copy()], torch.stack([t]), [mask_p.copy()], [emb_p], create_pred_xstart=True, **model_kwargs)
                pred_xstart = pred_xstart_outputs[0][2] # (target, output, pred_xstart)

                if self.decode_zooms:
                    pred_xstart_comp = decode_zooms(pred_xstart.copy(), sample_configs=sample_configs_p, out_zoom=max_zoom)
                else:
                    pred_xstart_comp = {max_zoom: pred_xstart[max_zoom]}

                self.log_tensor_plot(source_p, pred_xstart, target_p, mask_p, sample_configs_p, emb_p, self.current_epoch, output_comp=pred_xstart_comp, plot_name=f"_{t.item()}")

        return loss

    def predict_step(self, batch, batch_idx):
        # Call the desired parent's method directly
        # Note: Pass 'self' explicitly here
        return LightningProbabilisticModel.predict_step(self, batch, batch_idx)

    def _predict_step(self, source_groups, target_groups, patch_index_zooms, mask_groups, emb_groups):
        sample_configs = self.trainer.predict_dataloaders.dataset.sampling_zooms_collate or self.trainer.predict_dataloaders.dataset.sampling_zooms
        sample_configs = merge_sampling_dicts(sample_configs, patch_index_zooms)
        model_kwargs = {
            'sample_configs': sample_configs,
            'emb_groups': emb_groups
        }
        max_zoom = max(target_groups[0].keys()) if target_groups and target_groups[0] else None

        outputs = self.sampler.sample_loop(self.model, source_groups, mask_groups=mask_groups, progress=True, **model_kwargs)

        if self.decode_zooms:
            outputs = decode_zooms(outputs.copy(), sample_configs=sample_configs, out_zoom=max_zoom)
        return outputs
