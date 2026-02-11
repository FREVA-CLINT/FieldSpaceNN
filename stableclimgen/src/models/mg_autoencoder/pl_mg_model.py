from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
from pytorch_lightning.utilities import rank_zero_only
from ..mg_transformer.pl_mg_probabilistic import LightningProbabilisticModel
from ...models.mg_transformer.pl_mg_model import LightningMGModel, merge_sampling_dicts
from ...modules.grids.grid_utils import decode_zooms
from ...utils.helpers import merge_sampling_dicts


class LightningMGAutoEncoderModel(LightningMGModel, LightningProbabilisticModel):
    def __init__(
        self,
        model: Any,
        lr_groups: Mapping[str, Mapping[str, Any]],
        lambda_loss_dict: Dict[str, float],
        kl_weight: float = 1e-6,
        weight_decay: float = 0.0,
        n_samples: int = 1,
        max_batchsize: int = -1,
        mode: str = "encode_decode",
    ) -> None:
        """
        Initialize the Lightning wrapper for the multi-grid autoencoder.

        :param model: Autoencoder model instance.
        :param lr_groups: Optimizer parameter-group configuration.
        :param lambda_loss_dict: Loss weighting dictionary.
        :param kl_weight: KL divergence weight for probabilistic losses.
        :param weight_decay: Weight decay applied in the optimizer.
        :param n_samples: Number of posterior samples for probabilistic inference.
        :param max_batchsize: Optional cap on batch size during prediction.
        :param mode: Inference mode ("encode_decode", "encode", "decode").
        :return: None.
        """
        super().__init__(
            model,  # Main VAE model
            lr_groups,
            lambda_loss_dict=lambda_loss_dict,
            weight_decay=weight_decay
        )

        self.kl_weight: float = kl_weight
        self.n_samples: int = n_samples
        self.max_batchsize: int = max_batchsize
        self.mode: str = mode

    
    def training_step(
        self,
        batch: Tuple[Any, Any, Any, Any, Dict[int, torch.Tensor]],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Run one training step for the autoencoder.

        :param batch: Tuple ``(source_groups, target_groups, mask_groups, emb_groups, patch_index_zooms)``
            where tensors follow the base shape ``(b, v, t, n, d, f)``.
        :param batch_idx: Index of the current batch.
        :return: Training loss tensor.
        """
        sample_configs = self.trainer.val_dataloaders.dataset.sampling_zooms_collate or self.trainer.val_dataloaders.dataset.sampling_zooms
        source_groups, target_groups, mask_groups, emb_groups, patch_index_zooms = batch

        # Inject patch indices into the sampling configuration.
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
    

    def validation_step(
        self,
        batch: Tuple[Any, Any, Any, Any, Dict[int, torch.Tensor]],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Run one validation step for the autoencoder.

        :param batch: Tuple ``(source_groups, target_groups, mask_groups, emb_groups, patch_index_zooms)``.
        :param batch_idx: Index of the current batch.
        :return: Validation loss tensor.
        """
        sample_configs = self.trainer.val_dataloaders.dataset.sampling_zooms_collate or self.trainer.val_dataloaders.dataset.sampling_zooms
        source_groups, target_groups, mask_groups, emb_groups, patch_index_zooms = batch

        max_zooms = [max(target.keys()) for target in target_groups if target]
        max_zoom = max(max_zooms) if max_zooms else max(self.model.in_zooms)

        # Inject patch indices into the sampling configuration.
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

            # Decode outputs to the maximum zoom for visualization.
            output_comp = decode_zooms(output.copy(), sample_configs=sample_configs, out_zoom=max_zoom)

            self.logger.log_tensor_plot(source, output, target, mask, sample_configs, emb, max_zoom, self.current_epoch, output_comp=output_comp)

        return loss

    def predict_step(
        self,
        batch: Tuple[Any, Any, Any, Any, Dict[int, torch.Tensor]],
        batch_idx: int,
    ):
        """
        Run prediction using the probabilistic parent implementation.

        :param batch: Prediction batch tuple.
        :param batch_idx: Index of the current batch.
        :return: Prediction output dictionary.
        """
        # Call the desired parent's method directly
        # Note: Pass 'self' explicitly here
        return LightningProbabilisticModel.predict_step(self, batch, batch_idx)

    def _predict_step(
        self,
        source_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        target_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        patch_index_zooms: Dict[int, torch.Tensor],
        mask_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        emb_groups: Sequence[Dict[str, Any]],
    ):
        """
        Internal prediction step that supports encode/decode modes.

        :param source_groups: Source zoom-group inputs.
        :param target_groups: Target zoom-group inputs.
        :param patch_index_zooms: Patch indices per zoom.
        :param mask_groups: Mask groups aligned with inputs.
        :param emb_groups: Embedding groups aligned with inputs.
        :return: Output zoom-group mappings.
        """
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
