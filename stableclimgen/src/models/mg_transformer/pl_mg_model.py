import os
import math
import torch.nn as nn

import lightning.pytorch as pl
import torch
from typing import Dict
from torch.optim import AdamW

import mlflow
from lightning.pytorch.loggers import WandbLogger, MLFlowLogger

from pytorch_lightning.utilities import rank_zero_only
from ...modules.grids.grid_utils import decode_zooms

from ...utils.visualization import healpix_plot_zooms_var
from ...utils.losses import MSE_loss,GNLL_loss,NHVar_loss,NHTV_loss,NHInt_loss,L1_loss,NHTV_decay_loss


def check_empty(x):

    if isinstance(x, torch.Tensor):
        if x.numel()==0:
            return None
        else:
            return x      
    else:
        return x


def merge_sampling_dicts(sample_configs, patch_index_zooms):
    for key, value in patch_index_zooms.items():
        if key in sample_configs:
            sample_configs[key]['patch_index'] = value

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
        
        self.level_loss_fcns = {}   # {level: [{'lambda': x, 'fcn': ...}, ...]}
        self.common_loss_fcns = []  # [{'lambda': x, 'fcn': ...}, ...]

        for key, value in lambda_dict.items():
            if isinstance(key, (int, float)):  # Level-specific
                self.level_loss_fcns[key] = []
                for loss_name, lambda_ in value.items():
                    if float(lambda_) > 0:
                        self.level_loss_fcns[key].append({
                            'lambda': float(lambda_),
                            'fcn': globals()[loss_name](grid_layers[str(key)])
                        })
            else:  # Common loss
                if float(value) > 0:
                    self.common_loss_fcns.append({
                        'lambda': float(value),
                        'fcn': globals()[key](grid_layers[str(max_zoom)])
                    })

        self.has_elements = (len(self.level_loss_fcns) + len(self.common_loss_fcns)) > 0

    def forward(self, output, target, mask=None, sample_configs={}, prefix=''):
        loss_dict = {}
        total_loss = 0

        for level_key in output:
            out = output[level_key]
            tgt = target[level_key]

            # Level-specific losses
            if level_key in self.level_loss_fcns:
                for loss_fcn in self.level_loss_fcns[level_key]:
                    loss = loss_fcn['fcn'](out, tgt, mask=mask, sample_configs=sample_configs[level_key])
                    name = f"{prefix}level{level_key}_{loss_fcn['fcn']._get_name()}"
                    loss_dict[name] = loss.item()
                    total_loss += loss_fcn['lambda'] * loss

            # Common losses
            for loss_fcn in self.common_loss_fcns:
                loss = loss_fcn['fcn'](out, tgt, mask=mask, sample_configs=sample_configs[level_key])
                name = f"{prefix}level{level_key}_{loss_fcn['fcn']._get_name()}"
                loss_dict[name] = loss.item()
                total_loss += loss_fcn['lambda'] * loss

        return total_loss, loss_dict

class LightningMGModel(pl.LightningModule):
    def __init__(self, 
                 model, 
                 lr_groups, 
                 lambda_loss_dict: dict, 
                 weight_decay=0):
        
        super().__init__()

        self.model = model
        self.lr_groups = lr_groups  
        self.weight_decay = weight_decay
        
        self.save_hyperparameters(ignore=['model'])

        zooms_loss_dict = lambda_loss_dict.get("zooms",{})
        self.loss_zooms = MGMultiLoss(zooms_loss_dict, grid_layers=model.grid_layers, max_zoom=max(model.in_zooms))

        comp_loss_dict = lambda_loss_dict.get("composed",{})
        self.loss_composed = MGMultiLoss(comp_loss_dict, grid_layers=model.grid_layers, max_zoom=max(model.in_zooms))

    def forward(self, x, sample_configs={}, mask_zooms=None, emb=None, out_zoom=None) -> torch.Tensor:
        return self.model(x, sample_configs=sample_configs, mask_zooms=mask_zooms, emb=emb, out_zoom=out_zoom)
    
    def get_losses(self, source, target, sample_configs, mask_zooms=None, emb=None, prefix=''):
        
        loss_dict_total = {}
        total_loss = 0

        if self.loss_zooms.has_elements:
            output = self(source.copy(), sample_configs=sample_configs, mask_zooms=mask_zooms, emb=emb)

            loss, loss_dict = self.loss_zooms(output, target, mask=mask_zooms, sample_configs=sample_configs, prefix=f'{prefix}/')
            total_loss += loss
            loss_dict_total.update(loss_dict)
        else:
            output = None

        if self.loss_composed.has_elements:

            max_zoom = max(target.keys())
            target_comp = decode_zooms(target.copy(), sample_configs=sample_configs, out_zoom=max_zoom)
            output_comp = self(source.copy(), sample_configs=sample_configs, mask_zooms=mask_zooms, emb=emb, out_zoom=max_zoom)

            loss, loss_dict = self.loss_composed(output_comp, target_comp, mask=mask_zooms, sample_configs=sample_configs, prefix=f'{prefix}/composed_')
            total_loss += loss
            loss_dict_total.update(loss_dict)
        else:
            output_comp = None


        return total_loss, loss_dict_total, output, output_comp


    def training_step(self, batch, batch_idx):
        sample_configs = self.trainer.val_dataloaders.dataset.sampling_zooms_collate or self.trainer.val_dataloaders.dataset.sampling_zooms
        source, target, patch_index_zooms, mask, emb = batch

        sample_configs = merge_sampling_dicts(sample_configs, patch_index_zooms)

        loss, loss_dict, _, _ = self.get_losses(source, target, sample_configs, mask_zooms=mask, emb=emb, prefix='train')
      
        self.log_dict({"train/total_loss": loss.item()}, prog_bar=True)
        self.log_dict(loss_dict, logger=True)
        
        return loss
    


    def validation_step(self, batch, batch_idx):
        sample_configs = self.trainer.val_dataloaders.dataset.sampling_zooms_collate or self.trainer.val_dataloaders.dataset.sampling_zooms
        source, target, patch_index_zooms, mask, emb = batch

        max_zoom = max(target.keys())

        sample_configs = merge_sampling_dicts(sample_configs, patch_index_zooms)
        
        loss, loss_dict, output, output_comp = self.get_losses(source.copy(), target, sample_configs, mask_zooms=mask, emb=emb, prefix='val')

        self.log_dict({"validate/total_loss": loss.item()}, prog_bar=True)
        self.log_dict(loss_dict, logger=True)

        if batch_idx == 0 and rank_zero_only.rank==0:
            if output_comp is None:
                output_comp = self(source.copy(), sample_configs=sample_configs, mask_zooms=mask, emb=emb, out_zoom=max_zoom)
         
            self.log_tensor_plot(source, output, target, mask, sample_configs, emb, self.current_epoch, output_comp=output_comp)
            
        return loss


    def predict_step(self, batch, batch_idx):
        sample_configs = self.trainer.val_dataloaders.dataset.sampling_zooms_collate or self.trainer.val_dataloaders.dataset.sampling_zooms
        source, target, patch_index_zooms, mask, emb = batch

        sample_configs = merge_sampling_dicts(sample_configs, patch_index_zooms)
        output = self(source.copy(), sample_configs=sample_configs, mask=mask, emb=emb)

        output = self.model.decode(output, sample_configs, out_zoom=max(output.keys()), emb=emb)
        mask =self.model.decode(mask, sample_configs, out_zoom=max(mask.keys()), emb=emb)

        if hasattr(self.model,'predict_var') and self.model.predict_var:
            output, output_var = output.chunk(2, dim=-1)
        else:
            output_var = None

        output = {'output': output,
                'output_var': output_var,
                'mask': mask}
        return output
    

    def log_tensor_plot(self, input, output, gt, mask, sample_configs, emb, current_epoch, output_comp=None):

        save_dir = os.path.join(self.logger.save_dir if isinstance(self.logger, WandbLogger) else self.trainer.logger._tracking_uri, "validation_images")
        os.makedirs(save_dir, exist_ok=True)  

        save_paths_zooms = healpix_plot_zooms_var(input, output, gt, save_dir, mask_zooms=mask, sample_configs=sample_configs, plot_name=f"epoch_{current_epoch}", emb=emb)
    
        max_zoom = max(self.model.in_zooms)

        source_p = decode_zooms(input, sample_configs=sample_configs, out_zoom=max_zoom)
        target_p = decode_zooms(gt, sample_configs=sample_configs, out_zoom=max_zoom)

        if output_comp is None:
            output_p = self(input.copy(), sample_configs=sample_configs, mask_zooms=mask, emb=emb, out_zoom=max_zoom)
        else:
            output_p = output_comp

        mask = {max_zoom: mask[max_zoom]} if mask is not None else None
        save_paths = healpix_plot_zooms_var(source_p, output_p, target_p, save_dir, mask_zooms=mask, sample_configs=sample_configs, plot_name=f"epoch_{current_epoch}_combined", emb=emb)

        save_paths+=save_paths_zooms
        for k, save_path in enumerate(save_paths):
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


    def prepare_emb(self, emb=None, sample_configs={}):
        emb['CoordinateEmbedder'] = self.model.grid_layer_max.get_coordinates(**sample_configs)
        return emb