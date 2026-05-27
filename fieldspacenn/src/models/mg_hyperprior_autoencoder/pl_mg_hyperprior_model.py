"""PyTorch Lightning wrapper for the FieldSpace hyper-prior autoencoder."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import lightning.pytorch as pl
import torch
import torch.nn as nn
from pytorch_lightning.utilities import rank_zero_only

from ...modules.grids.grid_utils import decode_zooms
from ...modules.hyperprior.losses import HEALPixCRA5RateDistortionLoss
from ...utils.helpers import merge_sampling_dicts
from ...utils.schedulers import CosineWarmupScheduler


class LightningMGHyperpriorAutoEncoderModel(pl.LightningModule):
    """Lightning integration for ``MGHyperpriorFieldSpaceAutoEncoder``."""

    def __init__(
        self,
        model: Any,
        lr_groups: Mapping[str, Mapping[str, Any]],
        loss_config: Mapping[str, Any],
        weight_decay: float = 0.0,
        stage: str = "pretrain",
        freeze_analysis_encoder: bool = False,
        freeze_synthesis_decoder: bool = False,
        mode: str = "encode_decode",
        max_batchsize: int = -1,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr_groups = lr_groups
        self.weight_decay = float(weight_decay)
        self.stage = str(stage)
        self.mode = str(mode)
        self.max_batchsize = int(max_batchsize)
        self.save_hyperparameters(ignore=["model"])

        loss_kwargs = dict(loss_config or {})
        loss_kwargs.setdefault("grid_layers", model.grid_layers)
        self.loss = HEALPixCRA5RateDistortionLoss(**loss_kwargs)

        if freeze_analysis_encoder:
            self._freeze_modules(self.model.analysis_blocks, self.model.moment_projection)
        if freeze_synthesis_decoder:
            self._freeze_modules(self.model.synthesis_blocks, self.model.reconstruction_projection)

    def forward(
        self,
        x_zooms_groups: Optional[Sequence[Dict[int, torch.Tensor]]] = None,
        mask_zooms_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        emb_groups: Optional[Sequence[Dict[str, Any]]] = None,
        sample_configs: Mapping[int, Any] = {},
        out_zoom: Optional[int] = None,
        stage: Optional[str] = None,
        mask_zooms: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        emb: Optional[Sequence[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if x_zooms_groups is None:
            x_zooms_groups = []
        if isinstance(x_zooms_groups, dict):
            x_zooms_groups = [x_zooms_groups]
        if mask_zooms_groups is None:
            mask_zooms_groups = mask_zooms
        if emb_groups is None:
            emb_groups = emb
        return self.model(
            x_zooms_groups=x_zooms_groups,
            mask_zooms_groups=mask_zooms_groups,
            emb_groups=emb_groups,
            sample_configs=sample_configs,
            out_zoom=out_zoom,
            stage=stage or self.stage,
        )

    def get_losses(
        self,
        source_groups: Sequence[Dict[int, torch.Tensor]],
        target_groups: Sequence[Dict[int, torch.Tensor]],
        sample_configs: Mapping[int, Any] = {},
        mask_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        emb_groups: Optional[Sequence[Dict[str, Any]]] = None,
        prefix: str = "train",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]]:
        output = self(
            x_zooms_groups=[group.copy() for group in source_groups],
            mask_zooms_groups=mask_groups,
            emb_groups=emb_groups,
            sample_configs=sample_configs,
            stage=self.stage,
        )
        loss, loss_dict = self.loss(
            output,
            target_groups,
            mask_groups=mask_groups,
            sample_configs=sample_configs,
            emb_groups=emb_groups,
            model=self.model,
            prefix=prefix,
        )
        return loss, loss_dict, output

    def training_step(
        self,
        batch: Tuple[Any, Any, Any, Any, Dict[int, torch.Tensor]],
        batch_idx: int,
    ) -> torch.Tensor:
        source_groups, target_groups, mask_groups, emb_groups, patch_index_zooms = batch
        sample_configs = self._merged_sample_configs(patch_index_zooms, phase="fit")
        loss, loss_dict, _ = self.get_losses(
            source_groups,
            target_groups,
            sample_configs=sample_configs,
            mask_groups=mask_groups,
            emb_groups=emb_groups,
            prefix="train",
        )
        self.log("train/total_loss", loss, prog_bar=True)
        self.log_dict({key: value for key, value in loss_dict.items() if key != "train/total_loss"}, logger=True)
        return loss

    def validation_step(
        self,
        batch: Tuple[Any, Any, Any, Any, Dict[int, torch.Tensor]],
        batch_idx: int,
    ) -> torch.Tensor:
        source_groups, target_groups, mask_groups, emb_groups, patch_index_zooms = batch
        sample_configs = self._merged_sample_configs(patch_index_zooms, phase="validate")
        max_zooms = [max(target.keys()) for target in target_groups if target]
        max_zoom = max(max_zooms) if max_zooms else max(self.model.in_zooms)
        loss, loss_dict, output = self.get_losses(
            source_groups,
            target_groups,
            sample_configs=sample_configs,
            mask_groups=mask_groups,
            emb_groups=emb_groups,
            prefix="val",
        )
        self.log("val/total_loss", loss, prog_bar=True)
        self.log_dict({key: value for key, value in loss_dict.items() if key != "val/total_loss"}, logger=True)

        if batch_idx == 0 and rank_zero_only.rank == 0 and hasattr(self.logger, "log_healpix_tensor_plot"):
            output_groups = output["x_hat"]
            group_idx = next((idx for idx, group in enumerate(output_groups) if group), None)
            if group_idx is not None:
                output_group = output_groups[group_idx]
                output_comp = decode_zooms(output_group.copy(), sample_configs=sample_configs, out_zoom=max_zoom)
                self.logger.log_healpix_tensor_plot(
                    source_groups[group_idx],
                    output_group,
                    target_groups[group_idx],
                    mask_groups[group_idx] if mask_groups else None,
                    sample_configs,
                    emb_groups[group_idx] if emb_groups else {},
                    max_zoom,
                    self.current_epoch,
                    output_comp=output_comp,
                )
        return loss

    def predict_step(
        self,
        batch: Tuple[Any, Any, Any, Any, Dict[int, torch.Tensor]],
        batch_idx: int,
    ) -> Any:
        source_groups, target_groups, mask_groups, emb_groups, patch_index_zooms = batch
        return self._predict_step(source_groups, target_groups, patch_index_zooms, mask_groups, emb_groups)

    def _predict_step(
        self,
        source_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        target_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        patch_index_zooms: Dict[int, torch.Tensor],
        mask_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        emb_groups: Sequence[Dict[str, Any]],
    ) -> Any:
        sample_configs = self._merged_sample_configs(patch_index_zooms, phase="predict")
        max_zoom = max(self.model.in_zooms)
        if self.mode == "encode_decode":
            return self(
                x_zooms_groups=source_groups,
                mask_zooms_groups=mask_groups,
                emb_groups=emb_groups,
                sample_configs=sample_configs,
                out_zoom=max_zoom,
            )
        if self.mode == "encode":
            return self.model.ae_encode(source_groups, sample_configs=sample_configs, mask_groups=mask_groups, emb_groups=emb_groups)
        if self.mode == "decode":
            return self.model.ae_decode(source_groups, sample_configs=sample_configs, mask_groups=mask_groups, emb_groups=emb_groups, out_zoom=max_zoom)
        if self.mode == "compress":
            return self.model.compress(source_groups, sample_configs=sample_configs, mask_zooms_groups=mask_groups, emb_groups=emb_groups)
        if self.mode == "decompress":
            if not isinstance(source_groups, Mapping) or "strings" not in source_groups or "metadata" not in source_groups:
                raise ValueError("decompress prediction mode expects a mapping with 'strings' and 'metadata'.")
            return self.model.decompress(source_groups["strings"], source_groups["metadata"], sample_configs=sample_configs)
        raise ValueError(f"Unknown prediction mode '{self.mode}'.")

    def configure_optimizers(self):
        grouped_params = {group_name: [] for group_name in self.lr_groups}
        grouped_params.setdefault("default", [])
        seen_params = set()

        def visit_module(module: nn.Module) -> None:
            module_class_name = module.__class__.__name__
            matched_group = None
            for group_name, group_cfg in self.lr_groups.items():
                match_keys = group_cfg.get("matches", [group_name])
                if any(match_key in module_class_name for match_key in match_keys):
                    matched_group = group_name
                    break
            if matched_group is not None:
                for param in module.parameters(recurse=False):
                    if param.requires_grad and id(param) not in seen_params:
                        grouped_params[matched_group].append(param)
                        seen_params.add(id(param))
            for child in module.children():
                visit_module(child)

        visit_module(self)
        for param in self.parameters():
            if param.requires_grad and id(param) not in seen_params:
                grouped_params["default"].append(param)
                seen_params.add(id(param))

        param_groups = []
        for group_name, group_cfg in self.lr_groups.items():
            param_groups.append(
                {
                    "params": grouped_params[group_name],
                    "lr": group_cfg["lr"],
                    "name": group_name,
                    **{key: value for key, value in group_cfg.items() if key not in {"matches", "lr"}},
                }
            )
        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.weight_decay)
        max_steps = getattr(getattr(self, "trainer", None), "max_steps", 1000)
        scheduler = CosineWarmupScheduler(optimizer=optimizer, max_iters=max_steps, iter_start=0)
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def _merged_sample_configs(self, patch_index_zooms: Optional[Dict[int, torch.Tensor]], phase: str) -> Dict[int, Any]:
        sample_configs: Dict[int, Any] = {zoom: {} for zoom in self.model.in_zooms}
        try:
            loaders = {
                "fit": self.trainer.train_dataloader,
                "validate": self.trainer.val_dataloaders,
                "predict": self.trainer.predict_dataloaders,
            }
            loader = loaders.get(phase)
            dataset = getattr(loader, "dataset", None)
            sample_configs = (
                getattr(dataset, "sampling_zooms_collate", None)
                or getattr(dataset, "sampling_zooms", None)
                or sample_configs
            )
        except Exception:
            pass
        if patch_index_zooms:
            sample_configs = merge_sampling_dicts(sample_configs, patch_index_zooms)
        return sample_configs

    @staticmethod
    def _freeze_modules(*modules: nn.Module) -> None:
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
