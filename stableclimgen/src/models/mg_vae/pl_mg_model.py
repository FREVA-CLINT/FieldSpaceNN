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
                 mode="encode_decode"):
        
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

    
    def training_step(self, batch, batch_idx):
        sample_configs = self.trainer.val_dataloaders.dataset.sampling_zooms_collate or self.trainer.val_dataloaders.dataset.sampling_zooms
        source, target, patch_index_zooms, mask, emb = batch

        sample_configs = merge_sampling_dicts(sample_configs, patch_index_zooms)
        emb = self.prepare_emb(emb, sample_configs)

        loss, loss_dict, _, _, posterior_zooms = self.get_losses(source, target, sample_configs, mask_zooms=mask, emb=emb, prefix='train', post=True)

        # Compute KL divergence loss
        if self.kl_weight != 0.0:
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
        emb = self.prepare_emb(emb, sample_configs)

        loss, loss_dict, output, output_comp, posterior_zooms = self.get_losses(source, target, sample_configs, mask_zooms=mask,
                                                                 emb=emb, prefix='val', post=True)

        # Compute KL divergence loss
        if self.kl_weight != 0.0:
            kl_loss = torch.stack([posterior.kl() for posterior in posterior_zooms.values()]).mean(dim=0)
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
            loss = loss + self.kl_weight * kl_loss
            loss_dict['val/kl_loss'] = self.kl_weight * kl_loss

        self.log_dict({"val/total_loss": loss.item()}, prog_bar=True)
        self.log_dict(loss_dict, logger=True)

        if batch_idx == 0 and rank_zero_only.rank==0:
            if output_comp is None:
                output_comp, _ = self(source.copy(), sample_configs=sample_configs, mask_zooms=mask, emb=emb,
                                   out_zoom=max_zoom)

            self.log_tensor_plot(source, output, target,mask, sample_configs, emb, self.current_epoch, output_comp=output_comp)

        return loss

    def predict_step(self, batch, batch_idx):
        # Call the desired parent's method directly
        # Note: Pass 'self' explicitly here
        return LightningProbabilisticModel.predict_step(self, batch, batch_idx)

    def _predict_step(self, source, target, patch_index_zooms, mask, emb):
        sample_configs = self.trainer.val_dataloaders.dataset.sampling_zooms_collate or self.trainer.val_dataloaders.dataset.sampling_zooms

        sample_configs = merge_sampling_dicts(sample_configs, patch_index_zooms)
        output = self(source.copy(), sample_configs=sample_configs, mask=mask, emb=emb)

        if self.mode == "encode_decode":
            outputs, posterior = self.model(source, sample_configs=sample_configs, mask_zooms=mask, emb=emb)
        elif self.mode == "encode":
            outputs = self.model.vae_encode(source, sample_configs=sample_configs, mask_zooms=mask, emb=emb)
        elif self.mode == "decode":
            outputs = self.model.vae_decode(source, sample_configs=sample_configs, mask_zooms=mask, emb=emb)

        outputs = decode_zooms(outputs, max(outputs.keys()))
        return outputs