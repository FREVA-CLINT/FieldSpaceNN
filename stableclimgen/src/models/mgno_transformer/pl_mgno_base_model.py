import os
import math
import torch.nn as nn

import lightning.pytorch as pl
import torch

from torch.optim import AdamW

from pytorch_lightning.utilities import rank_zero_only
from ...utils.visualization import scatter_plot
from ...modules.grids.grid_layer import GridLayer, Interpolator


def check_empty(x):

    if isinstance(x, torch.Tensor):
        if x.numel()==0:
            return None
        else:
            return x      
    else:
        return x

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, lr_groups, max_iters, iter_start=0):
        self.max_num_iters = max_iters
        self.iter_start = iter_start
        self.warmups = [lr_group["lr_warmup"] for lr_group in lr_groups]
        self.zero_iters = [lr_group.get("zero_iters",0) for lr_group in lr_groups]
        super().__init__(optimizer)

    def get_lr(self):
        lr_factors = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factors[k] for k,base_lr in enumerate(self.base_lrs)]

    def get_lr_factor(self, epoch):
        epoch += self.iter_start
        lr_factors = [0.5 * (1 + math.cos(math.pi * epoch / self.max_num_iters)) for _ in range(len(self.warmups))]
        
        for k, lr_factor in enumerate(lr_factors):
            if epoch < self.zero_iters[k]:
                lr_factors[k] = 0
            elif epoch <= self.warmups[k]:
                lr_factors[k] = lr_factor*epoch * 1.0 / self.warmups[k]
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


        loss = ((output_nh[:,:,[0]] - output_nh[:,:,1:])).abs().mean().sqrt()
        return loss

class NH_loss_rel(nn.Module):
    def __init__(self, grid_layer):
        super().__init__()
        self.grid_layer: GridLayer = grid_layer

    def forward(self, output, target=None, sample_dict=None, mask=None,**kwargs):
        indices_in  = sample_dict["indices_layers"][int(self.grid_layer.global_zoom)] if sample_dict is not None and isinstance(sample_dict, dict) else None
        output_nh, _ = self.grid_layer.get_nh(output, indices_in, sample_dict)
        
        nh_rel = ((output_nh[:,:,[0]] - output_nh[:,:,1:])/output_nh[:,:,[0]]).abs()
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
                 noise_std=0.0, 
                 interpolator_settings=None):
        
        super().__init__()
        # maybe create multi_grid structure here?
        self.model = model

        if interpolator_settings is not None:
            self.interpolator = Interpolator(self.model.grid_layers,
                                             interpolator_settings.get("search_zoom_rel", 3),
                                             interpolator_settings.get("input_zoom", model.max_zoom),
                                             interpolator_settings.get("target_zoom", model.max_zoom),
                                             interpolator_settings.get("precompute", True),
                                             interpolator_settings.get("nh_inter", 3),
                                             interpolator_settings.get("power", 1),
                                             interpolator_settings.get("cutoff_dist_zoom", None),
                                             interpolator_settings.get("cutoff_dist", None),
                                             interpolator_settings.get("search_zoom_compute", None)
                                             )
        else:
            self.interpolator = None

        self.weight_decay = weight_decay
        self.lr_groups=lr_groups
        self.save_hyperparameters(ignore=['model'])
  
        self.noise_std = noise_std
        self.loss = MultiLoss(lambda_loss_dict, self.model.grid_layer_max if hasattr(self.model,'grid_layer_max') else None)

    def forward(self, x, coords_input, coords_output, sample_dict={}, mask=None, emb=None, dists_input=None):
        x, coords_input, coords_output, sample_dict, mask, emb, dists_input = self.prepare_inputs(x, coords_input, coords_output, sample_dict, mask, emb, dists_input)
        x: torch.tensor = self.model(x, coords_input=coords_input, coords_output=coords_output, sample_dict=sample_dict, mask=mask, emb=emb)
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

        if batch_idx == 0 and rank_zero_only:
            if hasattr(self, "interpolator") and self.interpolator is not None:
                input_inter, input_density = self.interpolator(source, mask=mask, input_coords=coords_input, sample_dict=sample_dict, calc_density=True, input_dists=rel_dists_input)
            else:
                input_inter = None
                input_density = None
            has_var = hasattr(self.model,'predict_var') and self.model.predict_var
            self.log_tensor_plot(source, output, target, coords_input, coords_output, mask, sample_dict,f"tensor_plot_{int(self.current_epoch)}", emb, input_inter=input_inter, input_density=input_density, has_var=has_var)
            

        return loss

    def get_coords_from_model(self, sample_dict={}):
        coords = self.model.grid_layers[str(self.model.max_zoom)].get_coordinates(**sample_dict)
        return coords


    def log_tensor_plot(self, input, output, gt, coords_input, coords_output, mask, sample_dict, plot_name, emb, input_inter=None, input_density=None, max_samples=8, has_var=False):

        save_dir = os.path.join(self.trainer.logger.save_dir, "validation_images")
        os.makedirs(save_dir, exist_ok=True)  
        
        if coords_input is None:
            coords_input = self.get_coords_from_model(sample_dict)

        if coords_output is None:
            coords_output = self.get_coords_from_model(sample_dict)

        sample = 0
        for k in range(input.shape[1]):
            var_idx = emb['VariableEmbedder'][sample,k]
            plot_name_var = f"{plot_name}_{var_idx}"
            save_path = os.path.join(save_dir, f"{plot_name_var}.png")
            
            if mask is not None:
                mask_p = mask[sample, k, :max_samples]
            else:
                mask_p = None

            if input_inter is not None:
                input_inter_p = input_inter[sample,k,:max_samples]
                input_density_p = input_density[sample,k,:max_samples]
            else:
                input_inter_p = None
                input_density_p = None

            scatter_plot(input[sample,k,:max_samples], 
                         output[sample,k,:max_samples], 
                         gt[sample,k,:max_samples], 
                         coords_input[sample] if coords_input.shape[0]>1 else coords_input[0], 
                         coords_output[sample] if coords_output.shape[0]>1 else coords_output[0], 
                         mask_p, 
                         input_inter=input_inter_p, 
                         input_density=input_density_p, 
                         save_path=save_path,
                         has_var=has_var)

            self.logger.log_image(f"plots/{plot_name_var}", [save_path])


    def configure_optimizers(self):
        no_params = []
        emb_params = []
        att_params = []
        att_params_names = []
        params = []
        for n, p in self.named_parameters():
            if 'sigma' in n or 'dist' in n:
                no_params.append(p)
            elif "attention" in n:
                att_params.append(p)
                att_params_names.append(n)
            else:
                params.append(p)
                    
        param_groups = []
        for lr_group in self.lr_groups:
            if lr_group['param_group']=='no_params':
                p = no_params
            elif lr_group['param_group']=='attention':
                p = att_params
            else:
                p = params
            param_groups.append({"params":p, "lr": lr_group["lr"], "name": lr_group['param_group']})


        optimizer = AdamW(param_groups, weight_decay=self.weight_decay)
        scheduler = CosineWarmupScheduler(
                        optimizer=optimizer,
                        lr_groups=self.lr_groups,
                        max_iters=self.trainer.max_steps,
                        iter_start=0)
        
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    
    def on_before_optimizer_step(self, optimizer):
        #for debug only
        pass
        # Check for parameters with no gradients before optimizer.step()
       # print("Checking for parameters with None gradients before optimizer step:")
        #for name, param in self.named_parameters():
         #   if param.grad is None:
          #      print(f"Parameter with no gradient: {name}")

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

        if self.interpolator:
            x, density_map = self.interpolator(x,
                                               mask=mask,
                                               calc_density=True,
                                               sample_dict=sample_dict,
                                               input_coords=coords_input,
                                               input_dists=dists_input)
            emb["DensityEmbedder"] = 1 - density_map
            emb["UncertaintyEmbedder"] = (density_map, emb['VariableEmbedder'])

            mask = None

        emb['CoordinateEmbedder'] = self.model.grid_layer_max.get_coordinates(**sample_dict)
        return x, coords_input, coords_output, sample_dict, mask, emb, dists_input