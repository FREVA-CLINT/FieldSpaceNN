import os
import math
import torch.nn as nn

import lightning.pytorch as pl
import torch
from typing import Any, Dict

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
from ...utils.losses import MSE_loss,GNLL_loss,NHVar_loss,NHTV_loss,NHInt_loss,L1_loss,NHTV_decay_loss,MSE_Hole_loss


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

        self.grid_layers = grid_layers or {}
        self.max_zoom = max_zoom

        # {level: {'default': [loss_entries], 'per_var': {var_id: {'label': str, 'losses': [loss_entries]}}}}
        self.level_loss_fcns: Dict[Any, Dict[str, Any]] = {}
        self.common_loss_fcns: Dict[str, Any] = {'default': [], 'per_var': {}}

        for key, value in lambda_dict.items():
            if self._is_level_key(key):
                level_key = self._normalize_level_key(key)
                level_entry = {'default': [], 'per_var': {}}
                default_spec, per_var_spec = self._split_loss_config(value)
                level_entry['default'].extend(self._build_losses(default_spec, self._get_grid_layer(level_key)))
                for var_key, loss_spec in per_var_spec.items():
                    losses = self._build_losses(loss_spec, self._get_grid_layer(level_key))
                    if not losses:
                        continue
                    normalized_var = self._normalize_var_key(var_key)
                    var_entry = level_entry['per_var'].setdefault(normalized_var, {'label': str(var_key), 'losses': []})
                    var_entry['losses'].extend(losses)
                if level_entry['default'] or level_entry['per_var']:
                    self.level_loss_fcns[level_key] = level_entry
            else:
                if isinstance(value, dict) and key not in globals():
                    default_spec, per_var_spec = self._split_loss_config(value)
                else:
                    default_spec, per_var_spec = self._split_loss_config({key: value})

                common_losses = self._build_losses(default_spec, self._get_grid_layer(self.max_zoom))
                self.common_loss_fcns['default'].extend(common_losses)

                for var_key, loss_spec in per_var_spec.items():
                    losses = self._build_losses(loss_spec, self._get_grid_layer(self.max_zoom))
                    if not losses:
                        continue
                    normalized_var = self._normalize_var_key(var_key)
                    var_entry = self.common_loss_fcns['per_var'].setdefault(normalized_var, {'label': str(var_key), 'losses': []})
                    var_entry['losses'].extend(losses)

        self.has_elements = self._compute_has_elements()

    def forward(self, output, target, mask=None, sample_configs={}, prefix='', emb=None):
        loss_dict = {}
        total_loss = 0.0

        group_embedder = None
        if isinstance(emb, dict) and isinstance(emb.get('GroupEmbedder'), torch.Tensor):
            group_embedder = emb['GroupEmbedder']

        for level_key in output:
            out = output[level_key]
            tgt = target[level_key]
            level_idx = self._normalize_level_key(level_key)

            mask_level = self._select_level_item(mask, level_key)
            mask_level = self._maybe_to_device(mask_level, out.device)

            sample_conf = self._select_level_item(sample_configs, level_key)

            level_losses = self.level_loss_fcns.get(level_idx)
            has_level_specific = level_losses is not None
            if level_losses is None:
                level_losses = {'default': [], 'per_var': {}}

            # Level-specific default losses
            for loss_fcn in level_losses['default']:
                loss = loss_fcn['fcn'](out, tgt, mask=mask_level, sample_configs=sample_conf)
                name = f"{prefix}level{level_idx}_{loss_fcn['fcn']._get_name()}"
                loss_dict[name] = loss.item()
                total_loss += loss_fcn['lambda'] * loss

            # Level-specific per-variable losses
            if group_embedder is not None and level_losses['per_var']:
                group_embedder_level = self._maybe_to_device(group_embedder, out.device)
                for var_id, var_entry in level_losses['per_var'].items():
                    var_mask = self._build_var_mask(group_embedder_level, var_id)
                    if var_mask is None or not torch.any(var_mask):
                        continue
                    out_var = out[var_mask]
                    tgt_var = tgt[var_mask]
                    mask_var = self._subset_mask(mask_level, var_mask)

                    for loss_fcn in var_entry['losses']:
                        loss = loss_fcn['fcn'](out_var, tgt_var, mask=mask_var, sample_configs=sample_conf)
                        name = f"{prefix}level{level_idx}_var{var_entry['label']}_{loss_fcn['fcn']._get_name()}"
                        loss_dict[name] = loss.item()
                        total_loss += loss_fcn['lambda'] * loss

            if not has_level_specific:
                # Common losses (default)
                for loss_fcn in self.common_loss_fcns['default']:
                    loss = loss_fcn['fcn'](out, tgt, mask=mask_level, sample_configs=sample_conf)
                    name = f"{prefix}level{level_idx}_{loss_fcn['fcn']._get_name()}"
                    loss_dict[name] = loss.item()
                    total_loss += loss_fcn['lambda'] * loss

                # Common per-variable losses
                if group_embedder is not None and self.common_loss_fcns['per_var']:
                    group_embedder_level = self._maybe_to_device(group_embedder, out.device)
                    for var_id, var_entry in self.common_loss_fcns['per_var'].items():
                        var_mask = self._build_var_mask(group_embedder_level, var_id)
                        if var_mask is None or not torch.any(var_mask):
                            continue
                        out_var = out[var_mask]
                        tgt_var = tgt[var_mask]
                        mask_var = self._subset_mask(mask_level, var_mask)

                        for loss_fcn in var_entry['losses']:
                            loss = loss_fcn['fcn'](out_var, tgt_var, mask=mask_var, sample_configs=sample_conf)
                            name = f"{prefix}level{level_idx}_var{var_entry['label']}_{loss_fcn['fcn']._get_name()}"
                            loss_dict[name] = loss.item()
                            total_loss += loss_fcn['lambda'] * loss

        return total_loss, loss_dict

    def _compute_has_elements(self):
        if self.common_loss_fcns['default'] or self.common_loss_fcns['per_var']:
            return True
        for level_entry in self.level_loss_fcns.values():
            if level_entry['default'] or level_entry['per_var']:
                return True
        return False

    def _is_level_key(self, key: Any) -> bool:
        if isinstance(key, (int, float)):
            return True
        if isinstance(key, str):
            stripped = key.strip()
            if stripped.startswith('-'):
                stripped = stripped[1:]
            return stripped.isdigit()
        try:
            int(str(key))
            return True
        except (TypeError, ValueError):
            return False

    def _normalize_level_key(self, key: Any):
        if isinstance(key, (int, float)):
            return int(key)
        if isinstance(key, str):
            stripped = key.strip()
            negative = stripped.startswith('-')
            if negative:
                stripped = stripped[1:]
            if stripped.isdigit():
                value = int(stripped)
                return -value if negative else value
        try:
            return int(str(key))
        except (TypeError, ValueError):
            return key

    def _normalize_var_key(self, key: Any):
        if isinstance(key, str):
            try:
                return int(key)
            except ValueError:
                return key
        if isinstance(key, (int, float)):
            return int(key)
        return key

    def _to_plain_dict(self, value):
        if DictConfig is not None and isinstance(value, DictConfig):
            return OmegaConf.to_container(value, resolve=True)
        return value

    def _split_loss_config(self, config):
        config = self._to_plain_dict(config)
        default_losses = {}
        per_var_losses = {}

        if not isinstance(config, dict):
            raise TypeError(f"Lambda configuration must be a dictionary, got {type(config)}")

        for cfg_key, cfg_val in config.items():
            cfg_val = self._to_plain_dict(cfg_val)
            if cfg_key == 'default' and isinstance(cfg_val, dict):
                default_losses.update(cfg_val)
                continue

            if cfg_key == 'per_var' and isinstance(cfg_val, dict):
                for var_key, loss_dict in cfg_val.items():
                    loss_dict = self._to_plain_dict(loss_dict)
                    if isinstance(loss_dict, dict):
                        per_var_losses.setdefault(var_key, {}).update(loss_dict)
                continue

            if isinstance(cfg_key, str) and cfg_key in globals():
                if isinstance(cfg_val, (int, float)):
                    default_losses[cfg_key] = cfg_val
                elif isinstance(cfg_val, dict):
                    if 'default' in cfg_val:
                        default_entry = self._to_plain_dict(cfg_val['default'])
                        if isinstance(default_entry, (int, float)):
                            default_losses[cfg_key] = default_entry
                    if 'per_var' in cfg_val:
                        per_var_entry = self._to_plain_dict(cfg_val['per_var'])
                        if isinstance(per_var_entry, dict):
                            for var_key, lambda_val in per_var_entry.items():
                                lambda_val = self._to_plain_dict(lambda_val)
                                if isinstance(lambda_val, dict):
                                    per_var_losses.setdefault(var_key, {}).update(lambda_val)
                                else:
                                    per_var_losses.setdefault(var_key, {})[cfg_key] = lambda_val
                    for var_key, lambda_val in cfg_val.items():
                        if var_key in ('default', 'per_var'):
                            continue
                        lambda_val = self._to_plain_dict(lambda_val)
                        if isinstance(lambda_val, (int, float)):
                            per_var_losses.setdefault(var_key, {})[cfg_key] = lambda_val
                        elif isinstance(lambda_val, dict):
                            per_var_losses.setdefault(var_key, {}).update(lambda_val)
                else:
                    raise ValueError(f"Invalid lambda specification for loss '{cfg_key}': expected number or dict, got {type(cfg_val)}")
            else:
                if isinstance(cfg_val, dict):
                    per_var_losses.setdefault(cfg_key, {}).update(cfg_val)
                else:
                    raise ValueError(f"Invalid lambda specification for key '{cfg_key}': expected dict, got {type(cfg_val)}")

        return default_losses, per_var_losses

    def _build_losses(self, loss_spec, grid_layer):
        losses = []
        if not isinstance(loss_spec, dict):
            return losses
        for loss_name, lambda_value in loss_spec.items():
            try:
                lambda_float = float(lambda_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Lambda for loss '{loss_name}' must be numeric, got {lambda_value}") from exc
            if lambda_float <= 0:
                continue
            if loss_name not in globals():
                raise KeyError(f"Unknown loss '{loss_name}' in lambda configuration")
            losses.append({
                'lambda': lambda_float,
                'fcn': globals()[loss_name](grid_layer=grid_layer)
            })
        return losses

    def _get_grid_layer(self, zoom):
        if self.grid_layers is None:
            return None
        if zoom is None:
            return None
        return self.grid_layers[str(zoom)]
    
    def _select_level_item(self, container, level_key):
        if level_key in container:
            return container[level_key]
        try:
            return container[int(level_key)]
        except (TypeError, ValueError, KeyError):
            pass
        return container.get(str(level_key), None)


    def _maybe_to_device(self, value, device):
        if torch.is_tensor(value):
            return value.to(device)
        return value

    def _build_var_mask(self, group_embedder, var_id):
        if group_embedder is None or not torch.is_tensor(group_embedder):
            return None
        if isinstance(var_id, str):
            try:
                var_id = int(var_id)
            except ValueError:
                return None
        if isinstance(var_id, (int, float)):
            return group_embedder == int(var_id)
        return None

    def _subset_mask(self, mask_level, var_mask):
        if mask_level is None:
            return None
        if torch.is_tensor(mask_level):
            if mask_level.numel() == 0:
                return mask_level
            return mask_level[var_mask]
        return mask_level

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

    def get_losses(self, source, target, sample_configs, mask_zooms=None, emb=None, prefix='', mode="default", pred_xstart=False):

        loss_dict_total = {}
        total_loss = 0
        posterior = None

        if self.loss_zooms.has_elements:
            if mode == "vae":
                output, posterior = self(source.copy(), sample_configs=sample_configs, mask_zooms=mask_zooms, emb=emb)
            elif mode == "diffusion":
                target, output, pred_xstart = self(source.copy(), sample_configs=sample_configs, mask_zooms=mask_zooms,
                                                   emb=emb, pred_xstart=pred_xstart)
            else:
                output = self(source.copy(), sample_configs=sample_configs, mask_zooms=mask_zooms, emb=emb)

            loss, loss_dict = self.loss_zooms(output, target, mask=mask_zooms, sample_configs=sample_configs, prefix=f'{prefix}/', emb=emb)
            total_loss += loss
            loss_dict_total.update(loss_dict)
        else:
            output = None

        if self.loss_composed.has_elements:

            max_zoom = max(target.keys())

            if mode == "vae":
                output_comp, posterior = self(source.copy(), sample_configs=sample_configs, mask_zooms=mask_zooms, emb=emb,
                                   out_zoom=max_zoom)
            elif mode == "diffusion":
                target, output_comp, pred_xstart_comp = self(source.copy(), sample_configs=sample_configs, mask_zooms=mask_zooms,
                                                             emb=emb, out_zoom=max_zoom, pred_xstart=pred_xstart)

            else:
                output_comp = self(source.copy(), sample_configs=sample_configs, mask_zooms=mask_zooms, emb=emb, out_zoom=max_zoom)

            target_comp = decode_zooms(target.copy(), sample_configs=sample_configs, out_zoom=max_zoom)
            loss, loss_dict = self.loss_composed(output_comp, target_comp, mask=mask_zooms, sample_configs=sample_configs, prefix=f'{prefix}/composed_', emb=emb)
            total_loss += loss
            loss_dict_total.update(loss_dict)
        else:
            output_comp = None

        if mode=="vae":
            return total_loss, loss_dict_total, output, output_comp, posterior
        elif mode=="diffusion":
            return total_loss, loss_dict_total, output, output_comp, pred_xstart
        else:
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
        sample_configs = self.trainer.predict_dataloaders.dataset.sampling_zooms_collate or self.trainer.predict_dataloaders.dataset.sampling_zooms
        source, target, patch_index_zooms, mask, emb = batch

        max_zoom = max(target.keys())
        sample_configs = merge_sampling_dicts(sample_configs, patch_index_zooms)

        output = self(source.copy(), sample_configs=sample_configs, mask_zooms=mask, emb=emb,
                           out_zoom=max_zoom)

        output = {'output': output,
                  'mask': mask}
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
            save_paths += healpix_plot_zooms_var(input, output, gt, save_dir, mask_zooms=mask, sample_configs=sample_configs, plot_name=f"epoch_{current_epoch}{plot_name}", emb=emb)
    
        max_zoom = max(self.model.in_zooms)

        source_p = decode_zooms(input, sample_configs=sample_configs, out_zoom=max_zoom)
        target_p = decode_zooms(gt, sample_configs=sample_configs, out_zoom=max_zoom)

        if output_comp is None:
            output_p = self(input.copy(), sample_configs=sample_configs, mask_zooms=mask, emb=emb, out_zoom=max_zoom)
        else:
            output_p = output_comp

        mask = {max_zoom: mask[max_zoom]} if mask is not None else None
        save_paths += healpix_plot_zooms_var(source_p, output_p, target_p, save_dir, mask_zooms=mask, sample_configs=sample_configs, plot_name=f"epoch_{current_epoch}_combined{plot_name}", emb=emb)

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

    #TODO fix
   # def prepare_emb(self, emb=None, sample_configs={}):
   #     emb['CoordinateEmbedder'] = (self.model.grid_layer_max.get_coordinates(**sample_configs[self.model.grid_layer_max.zoom]), emb["GroupEmbedder"])
   #     return emb
