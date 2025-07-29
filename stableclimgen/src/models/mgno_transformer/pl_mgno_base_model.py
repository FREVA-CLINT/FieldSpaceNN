import os
import math

import mlflow
import torch.nn as nn

import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers import WandbLogger, MLFlowLogger


from pytorch_lightning.utilities import rank_zero_only
from ...utils.visualization import healpix_plot_zooms_var
from ...modules.grids.grid_layer import GridLayer
from ...modules.grids.grid_utils import decode_zooms


def check_empty(x):

    if isinstance(x, torch.Tensor):
        if x.numel()==0:
            return None
        else:
            return x      
    else:
        return x

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

class L1_loss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.loss_fcn = torch.nn.SmoothL1Loss()

    def forward(self, output, target, **kwargs):
        loss = self.loss_fcn(output, target.view(output.shape))
        return loss

class MSE_loss(nn.Module):
    def __init__(self, grid_layer=None):
        super().__init__()

        self.loss_fcn = torch.nn.MSELoss() 

    def forward(self, output, target, **kwargs):
        loss = self.loss_fcn(output, target.view(output.shape))
        return loss
    
class GNLL_loss(nn.Module):
    def __init__(self, grid_layer=None):
        super().__init__()

        self.loss_fcn = torch.nn.GaussianNLLLoss() 

    def forward(self, output, target, **kwargs):
        output, output_var = output.chunk(2, dim=-1)
        loss = self.loss_fcn(output, target.view(*output.shape), output_var)
        return loss
    
class MSE_Hole_loss(nn.Module):
    def __init__(self, grid_layer=None):
        super().__init__()

        self.loss_fcn = torch.nn.MSELoss() 

    def forward(self, output, target, mask=None, sample_dict=None,**kwargs):
        if mask is not None:
            mask = mask.view(output.shape)
            target = target.view(output.shape)[mask]
            output = output[mask]
        loss = self.loss_fcn(output, target.view(output.shape))
        return loss

class MultiLoss(nn.Module):
    def __init__(self, lambda_dict, grid_layer=None):
        super().__init__()

        self.loss_fcns = []
        for target, lambda_ in lambda_dict.items():
            if float(lambda_) > 0:
                self.loss_fcns.append({'lambda': float(lambda_), 
                                        'fcn': globals()[target](grid_layer=grid_layer)})
    
    def forward(self, output, target, mask=None, sample_dict=None, prefix=''):
        loss_dict = {}
        total_loss = 0

        for loss_fcn in self.loss_fcns:
            loss = loss_fcn['fcn'](output, target, mask=mask, sample_dict=sample_dict)
            loss_dict[prefix + loss_fcn['fcn']._get_name()] = loss.item()
            total_loss = total_loss + loss_fcn['lambda'] * loss
        
        return total_loss, loss_dict


class Grad_loss(nn.Module):
    def __init__(self, grid_layer):
        super().__init__()
        self.grid_layer: GridLayer = grid_layer
        self.loss_fcn = torch.nn.KLDivLoss(log_target=True, reduction='batchmean') 

    def forward(self, output, target, sample_dict=None, mask=None,**kwargs):
        indices_in  = sample_dict["indices_layers"][int(self.grid_layer.global_zoom)] if sample_dict is not None and isinstance(sample_dict, dict) else None
        output_nh, _ = self.grid_layer.get_nh(output, indices_in, sample_dict)
        
        target_nh, _ = self.grid_layer.get_nh(target.view(output.shape), indices_in, sample_dict)

        nh_diff_output = 1+(output_nh[:,:,[0]] - output_nh[:,:,1:])/output_nh[:,:,[0]]
        nh_diff_target = 1+(target_nh[:,:,[0]] - target_nh[:,:,1:])/target_nh[:,:,[0]]

        nh_diff_output = nh_diff_output.view(output_nh.shape[0],-1).clamp(min=0, max=1)
        nh_diff_target = nh_diff_target.view(output_nh.shape[0],-1).clamp(min=0, max=1)

        nh_diff_output = nn.functional.log_softmax(nh_diff_output, dim=-1)
        nh_diff_target = nn.functional.log_softmax(nh_diff_target, dim=-1)

        loss = self.loss_fcn(nh_diff_output, nh_diff_target)
        return loss

class NH_loss(nn.Module):
    def __init__(self, grid_layer):
        super().__init__()
        self.grid_layer: GridLayer = grid_layer

    def forward(self, output, target=None, sample_dict={}, mask=None,**kwargs):
        output_nh, _ = self.grid_layer.get_nh(output, **sample_dict)


        loss = (((output_nh[...,[0],:] - output_nh[...,1:,:]))**2).mean().sqrt()
        return loss

class NH_loss_rel(nn.Module):
    def __init__(self, grid_layer):
        super().__init__()
        self.grid_layer: GridLayer = grid_layer

    def forward(self, output, target=None, sample_dict=None, mask=None,**kwargs):
        output_nh, _ = self.grid_layer.get_nh(output, **sample_dict)
        
        nh_rel = ((output_nh[...,[0],:] - output_nh[...,1:,:])/output_nh[...,[0],:]).abs()
        nh_rel = nh_rel[nh_rel>=torch.quantile(nh_rel, 0.98)]

        nh_rel = nh_rel.clamp(max=1)
        loss = (nh_rel).mean()
        return loss

class LightningMGNOBaseModel(pl.LightningModule):
    def __init__(self, 
                 model, 
                 lr_groups, 
                 lambda_loss_dict: dict, 
                 weight_decay=0, 
                 noise_std=0.0):
        
        super().__init__()
        # maybe create multi_grid structure here?
        self.model = model

        self.weight_decay = weight_decay
        self.lr_groups=lr_groups
        self.save_hyperparameters(ignore=['model'])
  
        self.noise_std = noise_std
        self.loss = MultiLoss(lambda_loss_dict, self.model.grid_layer_max if hasattr(self.model,'grid_layer_max') else None)

    def forward(self, x, coords_input, coords_output, sample_dict={}, mask=None, emb=None, dists_input=None):
        x, coords_input, coords_output, sample_dict, mask, emb, dists_input = self.prepare_inputs(x, coords_input, coords_output, sample_dict, mask, emb, dists_input)
        x: torch.tensor = self.model(x, coords_input=coords_input, coords_output=coords_output, sample_dict=sample_dict, mask_zooms=mask, emb=emb)
        return x

    def on_after_backward(self):
        if self.noise_std > 0:
            for param in self.model.parameters():
                if param.grad is not None:
                    noise = torch.randn_like(param.grad) * self.noise_std
                    param.grad += noise 
    

    def predict_step(self, batch, batch_idx):
        source, target, coords_input, coords_output, sample_dict, mask, emb, rel_dists_input, _ = batch
        output = self(source, coords_input=coords_input, coords_output=coords_output, sample_dict=sample_dict, mask=mask, emb=emb, dists_input=rel_dists_input)

        if hasattr(self.model,'predict_var') and self.model.predict_var:
            output, output_var = output.chunk(2, dim=-1)
        else:
            output_var = None

        output = {'output': output,
                'output_var': output_var,
                'mask': mask}
        return output
    

    def training_step(self, batch, batch_idx):
        source, target, coords_input, coords_output, sample_dict, mask, emb, rel_dists_input, _ = batch
        output = self(source, coords_input=coords_input, coords_output=coords_output, sample_dict=sample_dict, mask=mask, emb=emb, dists_input=rel_dists_input)
        
        loss, loss_dict = self.loss(output, target, mask=mask, sample_dict=sample_dict, prefix='train/')

        self.log_dict({"train/total_loss": loss.item()}, prog_bar=True)
        self.log_dict(loss_dict, logger=True)
        
        return loss


    def validation_step(self, batch, batch_idx):

        source, target, coords_input, coords_output, sample_dict, mask, emb, rel_dists_input, _ = batch

        coords_input, coords_output, mask, rel_dists_input = check_empty(coords_input), check_empty(coords_output), check_empty(mask), check_empty(rel_dists_input)
        sample_dict = self.prepare_sample_dict(sample_dict)

        output = self(source, coords_input=coords_input, coords_output=coords_output, sample_dict=sample_dict, mask=mask, emb=emb, dists_input=rel_dists_input)

        loss, loss_dict = self.loss(output, target, mask=mask, sample_dict=sample_dict, prefix='validate/')

        self.log_dict({"validate/total_loss": loss.item()}, prog_bar=True)
        self.log_dict(loss_dict, logger=True)

        if batch_idx == 0 and rank_zero_only.rank==0:
            has_var = hasattr(self.model,'predict_var') and self.model.predict_var
            self.log_tensor_plot(source, output, target, mask, sample_dict, emb, has_var=has_var)
            
        return loss


    def log_tensor_plot(self, input, output, gt, mask, sample_dict, emb, current_epoch):

        save_dir = os.path.join(self.logger.save_dir if isinstance(self.logger, WandbLogger) else self.trainer.logger._tracking_uri, "validation_images")
        os.makedirs(save_dir, exist_ok=True)  

        save_paths_zooms = healpix_plot_zooms_var(input, output, gt, save_dir, mask_zooms=mask, sample_dict=sample_dict, plot_name=f"epoch_{current_epoch}", emb=emb)
    
        max_zoom = max(self.model.in_zooms)

        source_p = {max_zoom: decode_zooms(input,max_zoom)}
        output_p = {max_zoom: decode_zooms(output,max_zoom)}
        target_p = {max_zoom: decode_zooms(gt,max_zoom)}
        mask = {max_zoom: mask[max_zoom]} if mask is not None else None
        save_paths = healpix_plot_zooms_var(source_p, output_p, target_p, save_dir, mask_zooms=mask, sample_dict=sample_dict, plot_name=f"epoch_{current_epoch}_combined", emb=emb)

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

    def prepare_sample_dict(self, sample_dict={}):
        if sample_dict is None:
            sample_dict = {}
        if 'zoom_patch_sample' in sample_dict.keys():
            sample_dict['zoom_patch_sample'] = sample_dict['zoom_patch_sample'][0] if sample_dict['zoom_patch_sample'].numel()>1 else sample_dict['zoom_patch_sample']
        return sample_dict

    def prepare_inputs(self, x, coords_input, coords_output, sample_dict={}, mask=None, emb=None, dists_input=None):
        coords_input, coords_output, mask, dists_input = check_empty(coords_input), check_empty(
            coords_output), check_empty(mask), check_empty(dists_input)
        sample_dict = self.prepare_sample_dict(sample_dict)


        emb['CoordinateEmbedder'] = self.model.grid_layer_max.get_coordinates(**sample_dict)
        return x, coords_input, coords_output, sample_dict, mask, emb, dists_input