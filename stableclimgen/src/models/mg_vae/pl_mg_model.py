import torch.nn as nn

import lightning.pytorch as pl
import torch
from typing import Dict

from pytorch_lightning.utilities import rank_zero_only


from ..mg_transformer.pl_mg_probabilistic import LightningProbabilisticModel
from ...modules.grids.grid_utils import decode_zooms
from ...models.mg_transformer.pl_mg_model import LightningMGModel,merge_sampling_dicts


class LightningMGVAEModel(LightningMGModel, LightningProbabilisticModel):
    def __init__(self, 
                 model, 
                 lr_groups,
                 lambda_loss_dict: dict,
                 kl_weight: float = 1e-6,
                 weight_decay=0,
                 decomposed_loss = True,
                 n_samples=1, max_batchsize=-1,
                 mode="encode_decode",
                 diff_input=True):
        
        super().__init__(
            model,  # Main VAE model
            lr_groups,
            lambda_loss_dict=lambda_loss_dict,
            weight_decay=weight_decay
        )

        self.kl_weight = kl_weight
        self.n_samples = n_samples
        self.max_batchsize = max_batchsize
        self.mode = mode
        self.diff_input = diff_input

    
    def training_step(self, batch, batch_idx):
        sample_configs = self.trainer.val_dataloaders.dataset.sampling_zooms_collate or self.trainer.val_dataloaders.dataset.sampling_zooms
        source, target, patch_index_zooms, mask, emb = batch

        sample_configs = merge_sampling_dicts(sample_configs, patch_index_zooms)
      #  emb = self.prepare_emb(emb, sample_configs)

        if not self.diff_input:
            new_source = self.model.decoder(source, emb=emb, sample_configs=sample_configs, out_zoom=max(target.keys()))
        else:
            new_source = source

        loss, loss_dict, _, posterior_zooms = self.get_losses(
            new_source,
            target,
            sample_configs,
            mask_zooms=mask,
            emb=emb,
            prefix='train',
            mode="vae",
        )

        # Compute KL divergence loss
        if self.kl_weight != 0.0 and self.model.distribution == "gaussian":
            kl_loss = torch.stack([posterior.kl() for posterior in posterior_zooms.values()]).mean(dim=0)
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
            loss = loss + self.kl_weight * kl_loss
            loss_dict['train/kl_loss'] = self.kl_weight * kl_loss

        self.log_dict({"train/total_loss": loss.item()}, prog_bar=True)
        self.log_dict(loss_dict, logger=True)

        return loss
    

    def validation_step(self, batch, batch_idx):
        sample_configs = self.trainer.val_dataloaders.dataset.sampling_zooms_collate or self.trainer.val_dataloaders.dataset.sampling_zooms
        source, target, patch_index_zooms, mask, emb = batch

        max_zoom = max(target.keys())

        sample_configs = merge_sampling_dicts(sample_configs, patch_index_zooms)

        if not self.diff_input:
            new_source = self.model.decoder(source, emb=emb, sample_configs=sample_configs, out_zoom=max_zoom)
        else:
            new_source = source

        loss, loss_dict, output_groups, posterior_zooms = self.get_losses(
            new_source,
            target,
            sample_configs,
            mask_zooms=mask,
            emb=emb,
            prefix='val',
            mode="vae",
        )
        output = output_groups[0] if isinstance(output_groups, list) else output_groups
        output_comp = None

        # Compute KL divergence loss
        if self.kl_weight != 0.0 and self.model.distribution == "gaussian":
            kl_loss = torch.stack([posterior.kl() for posterior in posterior_zooms.values()]).mean(dim=0)
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
            loss = loss + self.kl_weight * kl_loss
            loss_dict['val/kl_loss'] = self.kl_weight * kl_loss

        self.log_dict({"val/total_loss": loss.item()}, prog_bar=True)
        self.log_dict(loss_dict, logger=True)

        if batch_idx == 0 and rank_zero_only.rank==0:
            if output_comp is None:
                output_comp = self(
                    new_source.copy(),
                    sample_configs=sample_configs,
                    mask_zooms=mask,
                    emb=emb,
                    out_zoom=max_zoom,
                )

            self.log_tensor_plot(source, output, target,mask, sample_configs, emb, self.current_epoch, output_comp=output_comp)

        return loss

    def predict_step(self, batch, batch_idx):
        # Call the desired parent's method directly
        # Note: Pass 'self' explicitly here
        return LightningProbabilisticModel.predict_step(self, batch, batch_idx)

    def _predict_step(self, source, target, patch_index_zooms, mask, emb):
        sample_configs = self.trainer.predict_dataloaders.dataset.sampling_zooms_collate or self.trainer.predict_dataloaders.dataset.sampling_zooms
        sample_configs = merge_sampling_dicts(sample_configs, patch_index_zooms)

        if self.mode != "decode":
            source, sample_configs = self.prepare_missing_zooms(source, sample_configs)
            mask, _ = self.prepare_missing_zooms(mask)
            target, _ = self.prepare_missing_zooms(target)
        else:
            for zoom in self.model.in_zooms:
                sample_configs[zoom] = sample_configs[max(sample_configs.keys())]

        max_zoom = max(self.model.in_zooms)

        if self.mode == "encode_decode":
            outputs, posterior = self.model(source.copy(), sample_configs=sample_configs, mask_zooms=mask, emb=emb, out_zoom=max_zoom)
        elif self.mode == "encode":
            output_samples = self.model.vae_encode(source.copy(), sample_configs=sample_configs, mask_zooms=mask, emb=emb)
            outputs = {int(zoom): x.sample(gamma=self.model.gammas[str(zoom)] if self.model.gammas else None) for zoom, x in output_samples.items()}
        elif self.mode == "decode":
            outputs = self.model.vae_decode(source.copy(), sample_configs=sample_configs, mask_zooms=mask, emb=emb, out_zoom=max_zoom)

        return outputs
