import os
import math
import torch.nn as nn

import lightning.pytorch as pl
import torch
from typing import Any, Dict,List

try:
    from omegaconf import DictConfig, OmegaConf
except ImportError:  # pragma: no cover - OmegaConf might not be installed in some contexts
    DictConfig = None  # type: ignore
    OmegaConf = None  # type: ignore
from torch.optim import AdamW

import mlflow
from lightning.pytorch.loggers import WandbLogger, MLFlowLogger

from pytorch_lightning.utilities import rank_zero_only
from ...modules.grids.grid_utils import decode_zooms

from ...utils.visualization import healpix_plot_zooms_var
from ...utils.losses import MSE_masked_loss,MSE_loss,GNLL_loss,NHVar_loss,NHTV_loss,NHInt_loss,L1_loss,NHTV_decay_loss,MSE_Hole_loss


def check_empty(x):

    if isinstance(x, torch.Tensor):
        if x.numel()==0:
            return None
        else:
            return x      
    else:
        return x


def merge_sampling_dicts(sample_configs, patch_index_zooms):

    sample_configs = sample_configs.copy()

    for key, value in patch_index_zooms.items():
        if key in sample_configs.keys():
            sample_configs[key]['patch_index'] = value

    #TODO undirt
    for z in range(max(sample_configs.keys())):
        if z not in sample_configs.keys():
            sample_configs[z] = sample_configs[min(sample_configs.keys())]

    return sample_configs
    
class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_iters, iter_start=0):
        self.max_num_iters = max_iters
        self.iter_start = iter_start

        # Fetch per-group warmup and zero_iters from optimizer.param_groups
        self.warmups = [group.get("warmup", 1) for group in optimizer.param_groups]
        self.zero_iters = [group.get("zero_iters", 0) for group in optimizer.param_groups]

        super().__init__(optimizer)

    def get_lr(self):
        factor = self.get_lr_factors(self.last_epoch)
        return [base_lr * f for base_lr, f in zip(self.base_lrs, factor)]

    def get_lr_factors(self, epoch):
        epoch += self.iter_start
        lr_factors = [
            0.5 * (1 + math.cos(math.pi * epoch / self.max_num_iters))
            for _ in self.optimizer.param_groups
        ]

        for i in range(len(lr_factors)):
            if epoch < self.zero_iters[i]:
                lr_factors[i] = 0.0
            elif epoch <= self.warmups[i] and self.warmups[i]>0:
                lr_factors[i] *= epoch / (self.warmups[i])
            elif epoch <= self.warmups[i] and self.warmups[i]==0:
                lr_factors[i] *= epoch

        return lr_factors
    
class MGMultiLoss(nn.Module):
    def __init__(self, lambda_dict, grid_layers=None, max_zoom=None):
        super().__init__()
        self.grid_layers = grid_layers
        self.common_losses = nn.ModuleList()
        self.level_specific_losses = nn.ModuleDict()

        common_loss_config = lambda_dict.get("common", {})
        for loss_name, lambda_value in common_loss_config.items():
            self._add_loss(loss_name, lambda_value, self.common_losses)

        for key, level_config in lambda_dict.items():
            if str(key).isdigit():
                zoom_level = str(key)
                if zoom_level not in self.level_specific_losses:
                    self.level_specific_losses[zoom_level] = nn.ModuleList()
                for loss_name, lambda_value in level_config.items():
                    self._add_loss(loss_name, lambda_value, self.level_specific_losses[zoom_level], zoom_level)

    def _add_loss(self, loss_name, lambda_value, module_list, zoom_level=None):
        lambda_val = float(lambda_value)
        if lambda_val > 0:
            if loss_name not in globals():
                raise KeyError(f"Unknown loss '{loss_name}' in lambda configuration")

            grid_layer = self.grid_layers[zoom_level] if zoom_level is not None and self.grid_layers else None
            loss_instance = globals()[loss_name](grid_layer=grid_layer)
            
            # Store lambda and name on the instance for easy access in forward
            loss_instance.lambda_val = lambda_val
            loss_instance.loss_name = loss_name
            module_list.append(loss_instance)

    @property
    def has_elements(self):
        return len(self.common_losses) > 0 or any(len(v) > 0 for v in self.level_specific_losses.values())

    def forward(self, output, target, mask=None, sample_configs={}, prefix='', emb={}, lambda_group=0):
        loss_dict = {}
        total_loss = 0.0

        for zoom_level, out_zoom in output.items():
            tgt_zoom = target[zoom_level]
            mask_zoom = mask.get(zoom_level) if mask else None
            sample_conf = sample_configs.get(zoom_level) if sample_configs else None
            
            # Apply common losses
            for loss_fcn in self.common_losses:
                loss = loss_fcn(out_zoom, tgt_zoom, mask=mask_zoom, sample_configs=sample_conf)
                name = f"{prefix}level{zoom_level}_{loss_fcn._get_name()}"
                loss_dict[name] = lambda_group*loss.item()
                total_loss += lambda_group*loss_fcn.lambda_val * loss

            # Apply level-specific losses
            if str(zoom_level) in self.level_specific_losses:
                for loss_fcn in self.level_specific_losses[str(zoom_level)]:
                    loss = loss_fcn(out_zoom, tgt_zoom, mask=mask_zoom, sample_configs=sample_conf)
                    name = f"{prefix}level{zoom_level}_{loss_fcn._get_name()}"
                    loss_dict[name] = lambda_group*loss.item()
                    total_loss += lambda_group*loss_fcn.lambda_val * loss
        return total_loss, loss_dict


class LightningMGModel(pl.LightningModule):
    def __init__(self, 
                 model, 
                 lr_groups, 
                 lambda_loss_dict: Dict, 
                 lambda_loss_groups: List = None,
                 weight_decay=0,
                 n_autoregressive_steps=1):
        
        super().__init__()

        self.model = model
        self.lr_groups = lr_groups  
        self.weight_decay = weight_decay
        self.n_autoregressive_steps = n_autoregressive_steps
        
        self.save_hyperparameters(ignore=['model'])

        self.lambda_loss_groups = lambda_loss_groups
        zooms_loss_dict = lambda_loss_dict.get("zooms",{})
        self.loss_zooms = MGMultiLoss(zooms_loss_dict, grid_layers=model.grid_layers, max_zoom=max(model.in_zooms))

        comp_loss_dict = lambda_loss_dict.get("composed",{})
        self.loss_composed = MGMultiLoss(comp_loss_dict, grid_layers=model.grid_layers, max_zoom=max(model.in_zooms))


    def forward(self,  
                x_zooms_groups=None,
                mask_zooms_groups=None,
                emb_groups=None,
                sample_configs={},
                out_zoom=None,
                mask_zooms=None,
                emb=None,
                **kwargs) -> torch.Tensor:
        
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

    def get_losses(self, 
                   source_groups,
                   target_groups,
                   sample_configs={}, 
                   mask_groups=None,
                   emb_groups=None,
                   prefix='', 
                   mode="default", 
                   pred_xstart=False,
                   mask_zooms=None,
                   emb=None):

        loss_dict_total = {}
        total_loss = 0
        posterior = None

        if mask_groups is None:
            mask_groups = mask_zooms
        if emb_groups is None:
            emb_groups = emb

        if isinstance(source_groups, dict):
            source_groups = [source_groups]

        model_input = [group.copy() for group in source_groups]
        mask_input = mask_groups
        emb_input = emb_groups

        if mode == "diffusion":
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
            output_groups = self(
                x_zooms_groups=model_input,
                mask_zooms_groups=mask_input,
                emb_groups=emb_input,
                sample_configs=sample_configs,
            )

        lambda_groups = self.lambda_loss_groups if self.lambda_loss_groups is not None else [1]*len(source_groups)
        weight_groups = torch.tensor([list(t.values())[0].shape[1] for t in source_groups], device=emb_groups[0]['VariableEmbedder'].device)
        weight_groups = weight_groups/weight_groups.sum()

        for source, output, target, mask, emb, lambda_group, weight_group in zip(
            source_groups, output_groups, target_groups, mask_groups, emb_groups, lambda_groups, weight_groups
        ):
            if len(source) > 0:
                loss, loss_dict = self.loss_zooms(
                    output, target, mask=mask, sample_configs=sample_configs, prefix=f'{prefix}/', emb=emb, lambda_group=(lambda_group*weight_group)
                )
                total_loss += loss
                loss_dict_total.update(loss_dict)

        if self.loss_composed.has_elements:
            max_zooms = [max(target.keys()) for target in target_groups if target]
            if max_zooms:
                max_zoom = max(max_zooms)
                for source, output, target, mask, emb, lambda_group, weight_group in zip(
                    source_groups, output_groups, target_groups, mask_groups, emb_groups, lambda_groups, weight_groups
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
                            lambda_group=(lambda_group*weight_group)
                        )
                        total_loss += loss
                        loss_dict_total.update(loss_dict)


        if mode=="diffusion":
            return total_loss, loss_dict_total, output_groups, pred_xstart
        else:
            return total_loss, loss_dict_total, output_groups


    def get_losses_autoregressive(
        self,
        source_groups,
        target_groups,
        sample_configs={},
        mask_groups=None,
        emb_groups=None,
        prefix="",
        init_mode="repeat",
        n_autoregressive_steps=None,
        load_n_samples_time=None,
        mask_zooms=None,
        emb=None
    ):
        """
        Compute loss after rolling out autoregressively across consecutive samples.
        Expects batches where consecutive entries (per load_n_samples_time) represent consecutive timesteps.
        """
        if mask_groups is None:
            mask_groups = mask_zooms
        if emb_groups is None:
            emb_groups = emb

        time_embedder_all = [group['TimeEmbedder'].copy() for group in emb_groups]
        for group, emb in enumerate(emb_groups):
                for zoom, time_emb in emb['TimeEmbedder'].items():
                    emb_groups[group]['TimeEmbedder'][zoom] = time_emb[:,:-sample_configs[zoom].get('shift_n_ts_target',self.n_autoregressive_steps)]

        n_ar = n_autoregressive_steps or getattr(self.hparams, "n_autoregressive_steps", 1)
        if n_ar <= 1:
            return self.get_losses(
                source_groups,
                target_groups,
                sample_configs,
                mask_groups=mask_groups,
                emb_groups=emb_groups,
                prefix=prefix,
            )

        load_n_samples_time = self.trainer.datamodule.dataset_train.load_n_samples_time

        assert load_n_samples_time>1, "load_n_samples_time needs to be alrger than 1"

        b, t = source_groups[0][min(source_groups[0].keys())].shape[:2]

        output_groups = self(
                    x_zooms_groups=source_groups,
                    mask_zooms_groups=mask_groups,
                    emb_groups=emb_groups,
                    sample_configs=sample_configs
                    )
                
        for step in range(1, n_ar + 1):

            for group, output_zooms in enumerate(output_groups):
                for zoom, output in output_zooms.items():
                    output = torch.concat((output[:,:,1:], output[:,:,[-1]]), dim=2)
                    if self.trainer.datamodule.dataset_train.mask_ts_mode == 'zero':
                        output[:,:,-1] = 0

                    source_groups[group][zoom] = output

                if step < n_ar:
                    emb_groups[group]['TimeEmbedder'][zoom] = time_embedder_all[group][zoom][:,step:-(n_ar-step)]
                    output_groups = self(
                        x_zooms_groups=source_groups,
                        mask_zooms_groups=mask_groups,
                        emb_groups=emb_groups,
                        sample_configs=sample_configs
                        )
                else:
                    emb_groups[group]['TimeEmbedder'][zoom] = time_embedder_all[group][zoom][:,step:]
                    loss, loss_dict, output_groups = self.get_losses(
                        source_groups,
                        target_groups,
                        sample_configs,
                        mask_groups=mask_groups,
                        emb_groups=emb_groups,
                        prefix=prefix,
                    )

        return loss, loss_dict, output_groups

    def training_step(self, batch, batch_idx):
        sample_configs = self.trainer.val_dataloaders.dataset.sampling_zooms_collate or self.trainer.val_dataloaders.dataset.sampling_zooms
        source_groups, target_groups, mask_groups, emb_groups, patch_index_zooms = batch

        sample_configs = merge_sampling_dicts(sample_configs, patch_index_zooms)

        if getattr(self.hparams, "n_autoregressive_steps", 1) > 1:
            loss, loss_dict, _ = self.get_losses_autoregressive(
                source_groups,
                target_groups,
                sample_configs,
                mask_groups=mask_groups,
                emb_groups=emb_groups,
                prefix='train'
            )
        else:
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
    


    def validation_step(self, batch, batch_idx):
        sample_configs = self.trainer.val_dataloaders.dataset.sampling_zooms_collate or self.trainer.val_dataloaders.dataset.sampling_zooms
        source_groups, target_groups, mask_groups, emb_groups, patch_index_zooms = batch

        max_zooms = [max(target.keys()) for target in target_groups if target]
        max_zoom = max(max_zooms) if max_zooms else max(self.model.in_zooms)

        sample_configs = merge_sampling_dicts(sample_configs, patch_index_zooms)

   
        if getattr(self.hparams, "n_autoregressive_steps", 1) > 1:
            loss, loss_dict, output_groups = self.get_losses_autoregressive(
                [group.copy() for group in source_groups],
                target_groups,
                sample_configs=sample_configs,
                mask_groups=mask_groups,
                emb_groups=emb_groups,
                prefix='val'
            )
        else:
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

            self.log_tensor_plot(source, output, target, mask, sample_configs, emb, self.current_epoch, output_comp=output_comp)
            
        return loss


    def predict_step(self, batch, batch_idx):
        sample_configs = self.trainer.predict_dataloaders.dataset.sampling_zooms_collate or self.trainer.predict_dataloaders.dataset.sampling_zooms
        source_groups, target_groups, mask_groups, emb_groups, patch_index_zooms = batch

        max_zooms = [max(target.keys()) for target in target_groups if target]
        max_zoom = max(max_zooms) if max_zooms else max(self.model.in_zooms)
        sample_configs = merge_sampling_dicts(sample_configs, patch_index_zooms)

        output = self([group.copy() for group in source_groups], sample_configs=sample_configs, mask_zooms=mask_groups, emb=emb_groups,
                           out_zoom=max_zoom)

        output = {'output': output,
                  'mask': mask_groups}
        return output

    def prepare_missing_zooms(self, x_zooms, sample_configs=None):
        max_zoom = max(x_zooms.keys())
        for zoom in self.model.in_zooms:
            if zoom not in x_zooms.keys():
                x_zooms[zoom] = torch.zeros(1, 1, 1, 1, 1).expand(*x_zooms[max_zoom].shape[:3],
                                                                  int(x_zooms[max_zoom].shape[3] * 4**(zoom - max_zoom)),
                                                                  x_zooms[max_zoom].shape[4]).to(x_zooms[max_zoom].device)
                if sample_configs is not None:
                    sample_configs[zoom] = sample_configs[max_zoom]
        return x_zooms, sample_configs
    

    def log_tensor_plot(self, input, output, gt, mask, sample_configs, emb, current_epoch, output_comp=None, plot_name=""):

        save_dir = os.path.join(self.logger.save_dir if isinstance(self.logger, WandbLogger) else self.trainer.logger._tracking_uri, "validation_images")
        os.makedirs(save_dir, exist_ok=True)

        save_paths = []

        if output is not None:
            save_paths += healpix_plot_zooms_var(
                input,
                output,
                gt,
                save_dir,
                mask_zooms=mask,
                sample_configs=sample_configs,
                plot_name=f"epoch_{current_epoch}{plot_name}",
                emb=emb,
            )
    
        max_zoom = max(self.model.in_zooms)

        source_p = decode_zooms(input, sample_configs=sample_configs, out_zoom=max_zoom)
        target_p = decode_zooms(gt, sample_configs=sample_configs, out_zoom=max_zoom)

        if output_comp is None:
            output_p = self(
                input.copy(),
                sample_configs=sample_configs,
                mask_zooms=mask,
                emb=emb,
                out_zoom=max_zoom,
            )
        else:
            output_p = output_comp

        mask = {max_zoom: mask[max_zoom]} if mask is not None else None
        save_paths += healpix_plot_zooms_var(
            source_p,
            output_p,
            target_p,
            save_dir,
            mask_zooms=mask,
            sample_configs=sample_configs,
            plot_name=f"epoch_{current_epoch}_combined{plot_name}",
            emb=emb,
        )

        for k, save_path in enumerate(save_paths):
           if getattr(self,'log_images',False):
                if isinstance(self.logger, WandbLogger):
                    self.logger.log_image(f"plots/{os.path.basename(save_path).replace('.png','')}", [save_path])
                elif isinstance(self.logger, MLFlowLogger):
                    mlflow.log_artifact(save_path, artifact_path=f"plots/{os.path.basename(save_path).replace('.png','')}")


    def configure_optimizers(self):
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
    
    def on_before_optimizer_step(self, optimizer):
        #for debug only
        pass
        # Check for parameters with no gradients before optimizer.step()
       # print("Checking for parameters with None gradients before optimizer step:")
   #     for name, param in self.named_parameters():
   #         if param.grad is None:
   #             print(f"Parameter with no gradient: {name}")
