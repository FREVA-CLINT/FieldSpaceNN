
import torch
from pytorch_lightning.utilities import rank_zero_only
from ..mg_transformer.pl_mg_probabilistic import LightningProbabilisticModel
from ...models.mg_transformer.pl_mg_model import LightningMGModel,merge_sampling_dicts
from ...modules.grids.grid_utils import decode_zooms


class LightningMGAutoEncoderModel(LightningMGModel, LightningProbabilisticModel):
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
        source_groups, target_groups, mask_groups, emb_groups, patch_index_zooms = batch

        sample_configs = merge_sampling_dicts(sample_configs, patch_index_zooms)

        loss, loss_dict, _, = self.get_losses(
            source_groups,
            target_groups,
            sample_configs,
            mask_groups,
            emb_groups,
            prefix='val',
            mode="vae",
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

        loss, loss_dict, output_groups = self.get_losses(
            source_groups,
            target_groups,
            sample_configs,
            mask_groups,
            emb_groups,
            prefix='val',
            mode="vae",
        )
        output = output_groups[0] if isinstance(output_groups, list) else output_groups
        output_comp = None

        self.log_dict({"val/total_loss": loss.item()}, prog_bar=True)
        self.log_dict(loss_dict, logger=True)

        if batch_idx == 0 and rank_zero_only.rank==0:
            group_idx = next((idx for idx, group in enumerate(output_groups) if len(group) > 0), None)
            if group_idx is None:
                return loss

            output = output_groups[group_idx]
            source = source_groups[group_idx]
            target = target_groups[group_idx]
            mask = mask_groups[group_idx]
            emb = emb_groups[group_idx]

            output_comp = decode_zooms(output.copy(), sample_configs=sample_configs, out_zoom=max_zoom)

            self.log_tensor_plot(source, output, target, mask, sample_configs, emb, self.current_epoch, output_comp=output_comp)

        return loss

    def predict_step(self, batch, batch_idx):
        # Call the desired parent's method directly
        # Note: Pass 'self' explicitly here
        return LightningProbabilisticModel.predict_step(self, batch, batch_idx)

    def _predict_step(self, source_groups, target_groups, patch_index_zooms, mask_groups, emb_groups):
        sample_configs = self.trainer.predict_dataloaders.dataset.sampling_zooms_collate or self.trainer.predict_dataloaders.dataset.sampling_zooms
        sample_configs = merge_sampling_dicts(sample_configs, patch_index_zooms)

        processed_source_groups = []
        if self.mode != "decode":
            for group in source_groups:
                if group:
                    # The sample_configs are modified in-place by prepare_missing_zooms, so copy
                    processed_group, _ = self.prepare_missing_zooms(group.copy(), sample_configs.copy())
                    processed_source_groups.append(processed_group)
                else:
                    processed_source_groups.append(None)
        else:
            processed_source_groups = source_groups

        max_zoom = max(self.model.in_zooms)

        if self.mode == "encode_decode":
            # The self() call routes to the model's forward method, which does encode and decode.
            output_groups, _ = self(x_zooms_groups=processed_source_groups, sample_configs=sample_configs,
                                    mask_zooms_groups=mask_groups, emb_groups=emb_groups, out_zoom=max_zoom)
        elif self.mode == "encode":
            output_groups = self.model.ae_encode(processed_source_groups, sample_configs=sample_configs,
                                                 mask_groups=mask_groups, emb_groups=emb_groups)
        elif self.mode == "decode":
            output_groups = self.model.ae_decode(processed_source_groups, sample_configs=sample_configs,
                                                 mask_groups=mask_groups, emb_groups=emb_groups, out_zoom=max_zoom)
        else:
            raise ValueError(f"Unknown mode for prediction: {self.mode}")

        return output_groups
