import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, List

import lightning.pytorch as pl
import torch
import torch.nn as nn
from pytorch_lightning.utilities import rank_zero_only
from ...modules.grids.grid_utils import decode_zooms
from ...utils.losses import MGMultiLoss
from ...utils.schedulers import CosineWarmupScheduler
from ...utils.helpers import merge_sampling_dicts


class LightningMGModel(pl.LightningModule):
    def __init__(
        self,
        model: Any,
        lr_groups: Mapping[str, Mapping[str, Any]],
        lambda_loss_dict: Mapping[str, Any],
        weight_decay: float = 0.0,
        lambda_loss_groups: List = []
    ) -> None:
        """
        Initialize the Lightning wrapper for multi-grid transformer models.

        :param model: Multi-grid model instance.
        :param lr_groups: Optimizer parameter-group configuration.
        :param lambda_loss_dict: Loss weighting dictionary.
        :param weight_decay: Weight decay applied in the optimizer.
        :return: None.
        """
        
        super().__init__()

        self.model: Any = model
        self.lr_groups: Mapping[str, Mapping[str, Any]] = lr_groups  
        self.weight_decay: float = weight_decay
        
        self.save_hyperparameters(ignore=['model'])

        zooms_loss_dict = lambda_loss_dict.get("zooms",{})
        self.loss_zooms: MGMultiLoss = MGMultiLoss(
            zooms_loss_dict, grid_layers=model.grid_layers
        )

        comp_loss_dict = lambda_loss_dict.get("composed",{})
        self.loss_composed: MGMultiLoss = MGMultiLoss(
            comp_loss_dict, grid_layers=model.grid_layers
        )
        self.lambda_loss_groups = lambda_loss_groups


    def forward(
        self,
        x_zooms_groups: Optional[Sequence[Dict[int, torch.Tensor]]] = None,
        mask_zooms_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        emb_groups: Optional[Sequence[Dict[str, Any]]] = None,
        sample_configs: Mapping[int, Dict[str, Any]] = {},
        out_zoom: Optional[int] = None,
        mask_zooms: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        emb: Optional[Sequence[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Forward pass through the wrapped multi-grid model.

        :param x_zooms_groups: List of per-group zoom mappings with tensors of shape
            ``(b, v, t, n, d, f)``.
        :param mask_zooms_groups: Optional list of mask mappings aligned with inputs.
        :param emb_groups: Optional list of embedding dictionaries aligned with inputs.
        :param sample_configs: Sampling configuration dictionary per zoom.
        :param out_zoom: Optional target zoom level to decode outputs into.
        :param mask_zooms: Optional mask mappings used when ``mask_zooms_groups`` is None.
        :param emb: Optional embeddings used when ``emb_groups`` is None.
        :param kwargs: Additional arguments forwarded to the model.
        :return: Model outputs as zoom-group mappings.
        """
        
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
                out_zoom=out_zoom) 

    def get_losses(
        self,
        source_groups: Sequence[Dict[int, torch.Tensor]] | Dict[int, torch.Tensor],
        target_groups: Sequence[Dict[int, torch.Tensor]],
        sample_configs: Mapping[int, Dict[str, Any]] = {},
        mask_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        emb_groups: Optional[Sequence[Dict[str, Any]]] = None,
        prefix: str = '',
        mode: str = "default",
        pred_xstart: bool = False,
        mask_zooms: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        emb: Optional[Sequence[Dict[str, Any]]] = None,
    ):
        """
        Compute losses for a batch, optionally using diffusion mode.

        :param source_groups: Source zoom-group inputs with tensors of shape ``(b, v, t, n, d, f)``.
        :param target_groups: Target zoom-group inputs with tensors of shape ``(b, v, t, n, d, f)``.
        :param sample_configs: Sampling configuration dictionary per zoom.
        :param mask_groups: Optional mask groups aligned with inputs.
        :param emb_groups: Optional embedding groups aligned with inputs.
        :param prefix: Prefix for loss names.
        :param mode: Loss mode ("default" or "diffusion").
        :param pred_xstart: Whether to request ``x_0`` predictions in diffusion mode.
        :param mask_zooms: Optional mask groups used when ``mask_groups`` is None.
        :param emb: Optional embeddings used when ``emb_groups`` is None.
        :return: Tuple of ``(total_loss, loss_dict, output_groups[, pred_xstart])``.
        """

        loss_dict_total = {}
        total_loss = 0
        posterior = None

        if mask_groups is None:
            mask_groups = mask_zooms
        if emb_groups is None:
            emb_groups = emb

        source_is_dict = isinstance(source_groups, dict)
        if source_is_dict:
            source_groups_list = [source_groups]
        else:
            source_groups_list = list(source_groups)

        if source_is_dict:
            model_input = source_groups.copy()
            mask_input = mask_groups[0] if mask_groups else None
            emb_input = emb_groups[0] if emb_groups else None
        else:
            model_input = [group.copy() for group in source_groups_list]
            mask_input = mask_groups
            emb_input = emb_groups

        if mode == "diffusion":
            if source_is_dict:
                outputs = self(
                    model_input,
                    mask_zooms=mask_input,
                    emb=emb_input,
                    sample_configs=sample_configs,
                    pred_xstart=pred_xstart,
                )
            else:
                outputs = self(
                    x_zooms_groups=model_input,
                    mask_zooms_groups=mask_input,
                    emb_groups=emb_input,
                    sample_configs=sample_configs,
                    pred_xstart=pred_xstart,
                )

            # outputs is now a list of tuples, one for each group
            output_groups = []
            target_groups_from_diffusion = []
            pred_xstart_list = []
            for group_output in outputs:
                if group_output is not None and len(group_output) >= 2:
                    target_groups_from_diffusion.append(group_output[0])
                    output_groups.append(group_output[1])
                    pred_xstart_list.append(group_output[2] if len(group_output) > 2 else None)
            target_groups = target_groups_from_diffusion
            pred_xstart = pred_xstart_list[0] if pred_xstart_list else None # Keep single pred_xstart for now
            
        else:
            if source_is_dict:
                output_groups = self(
                    model_input,
                    mask_zooms=mask_input,
                    emb=emb_input,
                    sample_configs=sample_configs,
                )
            else:
                output_groups = self(
                    x_zooms_groups=model_input,
                    mask_zooms_groups=mask_input,
                    emb_groups=emb_input,
                    sample_configs=sample_configs,
                )

        lambda_groups = self.lambda_loss_groups if len(self.lambda_loss_groups)>0 else [1.0]*len(source_groups)
        weight_groups = torch.tensor([list(t.values())[0].shape[1] for t in source_groups], device=emb_groups[0]['VariableEmbedder'].device)
        weight_groups = weight_groups/weight_groups.sum()

        for source, output, target, mask, emb, lambda_group, weight_group in zip(
            source_groups_list, output_groups, target_groups, mask_groups, emb_groups, lambda_groups, weight_groups
        ):
        
            loss, loss_dict = self.loss_zooms(
                output, target, mask=mask, sample_configs=sample_configs, prefix=f'{prefix}/', emb=emb
            )
            total_loss += loss * (float(lambda_group) * weight_group)
            loss_dict_total.update(loss_dict)

        if self.loss_composed.has_elements:
            max_zooms = [max(target.keys()) for target in target_groups if target]
            if max_zooms:
                max_zoom = max(max_zooms)
                for source, output, target, mask, emb, lambda_group, weight_group in zip(
                    source_groups_list, output_groups, target_groups, mask_groups, emb_groups, lambda_groups, weight_groups
                ):
                    if len(source) > 0:
                        output_comp = decode_zooms(output.copy(), sample_configs=sample_configs, out_zoom=max_zoom)
                        target_comp = decode_zooms(target.copy(), sample_configs=sample_configs, out_zoom=max_zoom)

                        loss, loss_dict = self.loss_composed(
                            output_comp,
                            target_comp,
                            mask=mask,
                            sample_configs=sample_configs,
                            prefix=f'{prefix}/composed_',
                            emb=emb,
                        )
                        total_loss += loss * (float(lambda_group) * weight_group)
                        loss_dict_total.update(loss_dict)


        if mode=="diffusion":
            return total_loss, loss_dict_total, output_groups, pred_xstart
        else:
            return total_loss, loss_dict_total, output_groups

    def training_step(
        self,
        batch: Tuple[Any, Any, Any, Any, Dict[int, torch.Tensor]],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Run one training step for the multi-grid model.

        :param batch: Tuple ``(source_groups, target_groups, mask_groups, emb_groups, patch_index_zooms)``
            with tensors shaped ``(b, v, t, n, d, f)`` per zoom.
        :param batch_idx: Index of the current batch.
        :return: Training loss tensor.
        """
        sample_configs = self.trainer.val_dataloaders.dataset.sampling_zooms_collate or self.trainer.val_dataloaders.dataset.sampling_zooms
        source_groups, target_groups, mask_groups, emb_groups, patch_index_zooms = batch

        # Inject patch indices into the sampling configuration.
        sample_configs = merge_sampling_dicts(sample_configs, patch_index_zooms)

        loss, loss_dict, _ = self.get_losses(
            source_groups,
            target_groups,
            sample_configs,
            mask_groups=mask_groups,
            emb_groups=emb_groups,
            prefix='train',
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
        Run one validation step for the multi-grid model.

        :param batch: Tuple ``(source_groups, target_groups, mask_groups, emb_groups, patch_index_zooms)``
            with tensors shaped ``(b, v, t, n, d, f)`` per zoom.
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
            [group.copy() for group in source_groups],
            target_groups,
            sample_configs=sample_configs, 
            mask_groups=mask_groups,
            emb_groups=emb_groups,
            prefix='val')
        
        self.log_dict({"validate/total_loss": loss.item()}, prog_bar=True)
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

            self.logger.log_healpix_tensor_plot(source, output, target, mask, sample_configs, emb, max_zoom, self.current_epoch, output_comp=output_comp)
            
        return loss


    def predict_step(
        self,
        batch: Tuple[Any, Any, Any, Any, Dict[int, torch.Tensor]],
        batch_idx: int,
    ) -> Dict[str, Any]:
        """
        Run one prediction step for the multi-grid model.

        :param batch: Tuple ``(source_groups, target_groups, mask_groups, emb_groups, patch_index_zooms)``
            with tensors shaped ``(b, v, t, n, d, f)`` per zoom.
        :param batch_idx: Index of the current batch.
        :return: Dictionary with outputs and masks.
        """
        sample_configs = self.trainer.predict_dataloaders.dataset.sampling_zooms_collate or self.trainer.predict_dataloaders.dataset.sampling_zooms
        source_groups, target_groups, mask_groups, emb_groups, patch_index_zooms = batch

        max_zooms = [max(target.keys()) for target in target_groups if target]
        max_zoom = max(max_zooms) if max_zooms else max(self.model.in_zooms)
        sample_configs = merge_sampling_dicts(sample_configs, patch_index_zooms)

        output = self([group.copy() for group in source_groups], sample_configs=sample_configs, mask_zooms=mask_groups, emb=emb_groups,
                           out_zoom=max_zoom)

        output = {
            'output': output,
            'mask': mask_groups,
        }
        return output

    def prepare_missing_zooms(
        self,
        x_zooms: Dict[int, torch.Tensor],
        sample_configs: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> Tuple[Dict[int, torch.Tensor], Optional[Dict[int, Dict[str, Any]]]]:
        """
        Fill missing zoom levels with zero tensors to match the maximum zoom shape.

        :param x_zooms: Mapping from zoom to tensor of shape ``(b, v, t, n, d, f)``.
        :param sample_configs: Optional sampling configuration dictionary per zoom.
        :return: Updated ``x_zooms`` and ``sample_configs``.
        """
        max_zoom = max(x_zooms.keys())
        for zoom in self.model.in_zooms:
            if zoom not in x_zooms.keys():
                x_zooms[zoom] = torch.zeros(1, 1, 1, 1, 1).expand(*x_zooms[max_zoom].shape[:3],
                                                                  int(x_zooms[max_zoom].shape[3] * 4**(zoom - max_zoom)),
                                                                  x_zooms[max_zoom].shape[4]).to(x_zooms[max_zoom].device)
                if sample_configs is not None:
                    sample_configs[zoom] = sample_configs[max_zoom]
        return x_zooms, sample_configs


    def configure_optimizers(self):
        """
        Build optimizer and cosine warmup scheduler.

        :return: Optimizer and scheduler configuration for Lightning.
        """
        grouped_params = {group_name: [] for group_name in self.lr_groups}
        grouped_params["default"] = []  # fallback group
        seen_params = set()
        
        class_names = []
        def visit_module(module):
            # Match this module by class name
            module_class_name = module.__class__.__name__
            class_names.append(module_class_name)
            matched = False
            for group_name, group_cfg in self.lr_groups.items():
                match_keys = group_cfg.get("matches", [group_name])
                if any(mk in module_class_name for mk in match_keys):
                    matched = True
                    break

            if matched:
                for p in module.parameters():
                    if id(p) not in seen_params and p.requires_grad:
                        grouped_params[group_name].append(p)
                        seen_params.add(id(p))

            # Recurse into submodules (including inside ModuleList/Dict)
            for name, child in module._modules.items():
                if isinstance(child, (nn.ModuleList, nn.ModuleDict)):
                    for sub_child in child.values() if isinstance(child, nn.ModuleDict) else child:
                        visit_module(sub_child)
                elif isinstance(child, nn.Module):
                     visit_module(child)

        # Start recursive traversal from self (i.e. the whole model)
        visit_module(self)

        # Assign leftover parameters to default group
        for p in self.parameters():
            if id(p) not in seen_params and p.requires_grad:
                grouped_params["default"].append(p)
                seen_params.add(id(p))

        param_groups = []
        for group_name, group_cfg in self.lr_groups.items():
            param_groups.append({
                "params": grouped_params[group_name],
                "lr": group_cfg["lr"],
                "name": group_name,
                **{k: v for k, v in group_cfg.items() if k not in {"matches", "lr"}}
            })

        optimizer = torch.optim.Adam(param_groups, weight_decay=self.weight_decay)

        scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            max_iters=self.trainer.max_steps,
            iter_start=0
        )
        
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
    
    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """
        Hook executed before optimizer step (kept for debugging).

        :param optimizer: Optimizer instance about to step.
        :return: None.
        """
        #for debug only
        pass
        # Check for parameters with no gradients before optimizer.step()
       # print("Checking for parameters with None gradients before optimizer step:")
   #     for name, param in self.named_parameters():
   #         if param.grad is None:
   #             print(f"Parameter with no gradient: {name}")
