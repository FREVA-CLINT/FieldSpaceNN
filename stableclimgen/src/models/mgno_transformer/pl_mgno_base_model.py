import os
import math
import torch.nn as nn

import lightning.pytorch as pl
import torch

from torch.optim import AdamW

from ...utils.visualization import scatter_plot
from pytorch_lightning.utilities import rank_zero_only

from ...modules.icon_grids.grid_layer import GridLayer, Interpolator


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


class MSE_loss(nn.Module):
    def __init__(self, grid_layer=None):
        super().__init__()

        self.loss_fcn = torch.nn.MSELoss() 

    def forward(self, output, target, mask=None, indices_sample=None,**kwargs):
        loss = self.loss_fcn(output, target.view(output.shape))
        return loss
    
class GNLLL_loss(nn.Module):
    def __init__(self, grid_layer=None):
        super().__init__()

        self.loss_fcn = torch.nn.GaussianNLLLoss() 

    def forward(self, output, target, output_var=None,**kwargs):
        loss = self.loss_fcn(output, target.view(*output.shape), output_var)
        return loss
    
class MSE_Hole_loss(nn.Module):
    def __init__(self, grid_layer=None):
        super().__init__()

        self.loss_fcn = torch.nn.MSELoss() 

    def forward(self, output, target, mask=None, indices_sample=None,**kwargs):
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
            if lambda_ > 0:
                self.loss_fcns.append({'lambda': float(lambda_), 
                                        'fcn': globals()[target](grid_layer=grid_layer)})
    
    def forward(self, output, target, mask=None, indices_sample=None, prefix=''):
        loss_dict = {}
        total_loss = 0
        if output.shape[-1]>1:
            output_var = output[...,1]
            output = output[...,0]
        else:
            output_var = None

        for loss_fcn in self.loss_fcns:
            loss = loss_fcn['fcn'](output, target, mask=mask, indices_sample=indices_sample, output_var=output_var)
            loss_dict[prefix + loss_fcn['fcn']._get_name()] = loss.item()
            total_loss = total_loss + loss_fcn['lambda'] * loss
        
        return total_loss, loss_dict


class Grad_loss(nn.Module):
    def __init__(self, grid_layer):
        super().__init__()
        self.grid_layer: GridLayer = grid_layer
        self.loss_fcn = torch.nn.KLDivLoss(log_target=True, reduction='batchmean') 

    def forward(self, output, target, indices_sample=None, mask=None,**kwargs):
        indices_in  = indices_sample["indices_layers"][int(self.grid_layer.global_level)] if indices_sample is not None and isinstance(indices_sample, dict) else None
        output_nh, _ = self.grid_layer.get_nh(output, indices_in, indices_sample)
        
        target_nh, _ = self.grid_layer.get_nh(target.view(output.shape), indices_in, indices_sample)

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

    def forward(self, output, target=None, indices_sample=None, mask=None,**kwargs):
        indices_in  = indices_sample["indices_layers"][int(self.grid_layer.global_level)] if indices_sample is not None and isinstance(indices_sample, dict) else None
        output_nh, _ = self.grid_layer.get_nh(output, indices_in, indices_sample)

        loss = ((output_nh[:,:,[0]] - output_nh[:,:,1:])).abs().mean().sqrt()
        return loss

class NH_loss_rel(nn.Module):
    def __init__(self, grid_layer):
        super().__init__()
        self.grid_layer: GridLayer = grid_layer

    def forward(self, output, target=None, indices_sample=None, mask=None,**kwargs):
        indices_in  = indices_sample["indices_layers"][int(self.grid_layer.global_level)] if indices_sample is not None and isinstance(indices_sample, dict) else None
        output_nh, _ = self.grid_layer.get_nh(output, indices_in, indices_sample)
        
        nh_rel = ((output_nh[:,:,[0]] - output_nh[:,:,1:])/output_nh[:,:,[0]]).abs()
        nh_rel = nh_rel[nh_rel>=torch.quantile(nh_rel, 0.98)]

        nh_rel = nh_rel.clamp(max=1)
        loss = (nh_rel).mean()
        return loss

class LightningMGNOBaseModel(pl.LightningModule):
    def __init__(self, model, lr_groups, lambda_loss_dict: dict, weight_decay=0, noise_std=0.0, interpolator_settings=None):
        super().__init__()
        # maybe create multi_grid structure here?
        self.model = model

        if interpolator_settings is not None:
            self.interpolator = Interpolator(self.model.grid_layers,
                                             interpolator_settings.get("search_level", 3),
                                             interpolator_settings.get("input_level", 0),
                                             interpolator_settings.get("target_level", 0),
                                             interpolator_settings.get("precompute", True),
                                             interpolator_settings.get("nh_inter", 3),
                                             interpolator_settings.get("power", 1),
                                             interpolator_settings.get("cutoff_dist_level", None),
                                             interpolator_settings.get("cutoff_dist", None),
                                             interpolator_settings.get("search_level_compute", None)
                                             )
        else:
            self.interpolator = None

        self.weight_decay = weight_decay
        self.lr_groups=lr_groups
        self.save_hyperparameters(ignore=['model'])
  
        self.noise_std = noise_std
        self.loss = MultiLoss(lambda_loss_dict, self.model.grid_layer_0)

    def forward(self, x, coords_input, coords_output, indices_sample=None, mask=None, emb=None, dists_input=None):
        b, nt, n = x.shape[:3]
        coords_input, coords_output, indices_sample, mask, emb, dists_input = check_empty(coords_input), check_empty(coords_output), check_empty(indices_sample), check_empty(mask), check_empty(emb), check_empty(dists_input)
        x, coords_input, coords_output, indices_sample, mask, emb, dists_input = self.prepare_batch(x, coords_input, coords_output, indices_sample, mask, emb, dists_input)
        indices_sample, coords_input, coords_output, dists_input = self.prepare_coords_indices(coords_input,
                                                                                               coords_output=coords_output,
                                                                                               indices_sample=indices_sample,
                                                                                               input_dists=dists_input)
        if self.interpolator:
            x, density_map = self.interpolator(x,
                                  mask=mask,
                                  calc_density=True,
                                  indices_sample=indices_sample,
                                  input_coords=coords_input,
                                  input_dists=dists_input)
            emb["DensityEmbedder"] = 1 - density_map.transpose(-2, -1)
            emb["UncertaintyEmbedder"] = (density_map.transpose(-2, -1), emb['VariableEmbedder'])

            mask = None
        x: torch.tensor = self.model(x, coords_input=coords_input, coords_output=coords_output, indices_sample=indices_sample, mask=mask, emb=emb)
        return x.view(b, nt, *x.shape[1:])

    def on_after_backward(self):
        if self.noise_std > 0:
            for param in self.model.parameters():
                if param.grad is not None:
                    noise = torch.randn_like(param.grad) * self.noise_std
                    param.grad += noise 
    

    def predict_step(self, batch, batch_idx):
        source, target, coords_input, coords_output, indices, mask, emb, dists_input = batch
        output = self(source, coords_input=coords_input, coords_output=coords_output, indices_sample=indices, mask=mask, emb=emb, dists_input=dists_input)
        output = {'output': output[...,0].unsqueeze(dim=1),
                  'output_var': output[...,1].unsqueeze(dim=1) if hasattr(self.model,'predict_var') and self.model.predict_var else None,
                  'mask': mask.unsqueeze(dim=1)}
        return output
    

    def training_step(self, batch, batch_idx):
        source, target, coords_input, coords_output, indices, mask, emb, dists_input = batch
        output = self(source, coords_input=coords_input, coords_output=coords_output, indices_sample=indices, mask=mask, emb=emb, dists_input=dists_input)
        
        loss, loss_dict = self.loss(output, target, mask=mask, indices_sample=indices, prefix='train/')

        self.log_dict({"train/total_loss": loss.item()}, prog_bar=True)
        self.log_dict(loss_dict, logger=True)
        
        return loss


    def validation_step(self, batch, batch_idx):
        source, target, coords_input, coords_output, indices, mask, emb, dists_input = batch
        coords_input, coords_output, indices, mask, emb, dists_input = check_empty(coords_input), check_empty(coords_output), check_empty(indices), check_empty(mask), check_empty(emb), check_empty(dists_input)
        output = self(source, coords_input=coords_input, coords_output=coords_output, indices_sample=indices, mask=mask, emb=emb, dists_input=dists_input)

        loss, loss_dict = self.loss(output, target, mask=mask, indices_sample=indices, prefix='validate/')

        self.log_dict({"validate/total_loss": loss.item()}, prog_bar=True)
        self.log_dict(loss_dict, logger=True)

        if batch_idx == 0 and rank_zero_only:
            if hasattr(self, "interpolator") and self.interpolator is not None:
                _, coords_input, _, _, _, _, dists_input = self.prepare_batch(source, coords_input=coords_input, input_dists=dists_input)
                input_inter, input_density = self.interpolator(source, mask=mask, input_coords=coords_input, indices_sample=indices, calc_density=True, input_dists=dists_input)
            else:
                input_inter = None
                input_density = None
            self.log_tensor_plot(source, output, target, coords_input, coords_output, mask, indices,f"tensor_plot_{self.current_epoch}", emb, input_inter=input_inter, input_density=input_density)
            

        return loss

    def get_coords_from_model(self, indices_dict=None):
        if indices_dict is not None and isinstance(indices_dict, dict):
            indices = self.model.get_global_indices_local(indices_dict['sample'], 
                                                            indices_dict['sample_level'], 
                                                            0)
            coords = self.model.cell_coords_global[indices].unsqueeze(dim=-2)
        else:
            coords = self.model.cell_coords_global.unsqueeze(dim=0).unsqueeze(dim=-2)
        return coords


    def log_tensor_plot(self, input, output, gt, coords_input, coords_output, mask, indices_dict, plot_name, emb, input_inter=None, input_density=None, max_samples=8):

        save_dir = os.path.join(self.trainer.logger.save_dir, "validation_images")
        os.makedirs(save_dir, exist_ok=True)  
        
        if coords_input is None:
            coords_input = self.get_coords_from_model(indices_dict)

        if coords_input.shape[0] != 1:
            coords_input = coords_input.view(input.shape[0], input.shape[1], *coords_input.shape[1:])

        elif len(coords_input.shape) == 4:
            coords_input = coords_input.repeat(input.shape[1], 1, 1, 1).unsqueeze(0)

        if coords_output is None:
            coords_output = self.get_coords_from_model(indices_dict)

        if coords_output.shape[0] != 1:
            coords_output = coords_output.view(input.shape[0], input.shape[1], *coords_output.shape[1:])

        elif len(coords_output.shape) == 4:
            coords_output = coords_output.repeat(input.shape[1], 1, 1, 1).unsqueeze(0)

        sample = 0
        for k in range(input.shape[-2]):
            var_idx = emb['VariableEmbedder'][sample, k].item()
            plot_name_var = f"{plot_name}_{var_idx}"
            save_path = os.path.join(save_dir, f"{plot_name_var}.png")
            
            if mask is not None:
                mask_p = mask[sample, :max_samples, :, :, k]
            else:
                mask_p = None

            if input_inter is not None:
                input_inter_p = input_inter[sample,:max_samples,:,k]
                input_density_p = input_density[sample,:max_samples,:,:,k]
            else:
                input_inter_p = None
                input_density_p = None

            scatter_plot(input[sample,:max_samples,:,:,k], output[sample,:max_samples,:,k], gt[sample,:max_samples,:,:,k], coords_input[sample, :max_samples], coords_output[sample, :max_samples], mask_p, input_inter=input_inter_p, input_density=input_density_p, save_path=save_path)

            self.logger.log_image(f"plots/{plot_name_var}", [save_path])


    def get_no_params_dict(self):
        param_dict = {}
        for block_idx, block in enumerate(self.model.Blocks):
            for  no_block in block.NO_Blocks:
                for sigma_idx, sigma in enumerate(no_block.no_layer.sigma):
                    param_dict[f'no_lvl_{no_block.no_layer.grid_layer_no.global_level.item()}/sigma_{sigma_idx}/block_{block_idx}'] = sigma.item()
                for dist_idx, dist in enumerate(no_block.no_layer.dists):
                    param_dict[f'no_lvl_{no_block.no_layer.grid_layer_no.global_level.item()}/dist_{dist_idx}/block_{block_idx}'] = dist.item()
            
        return param_dict
    
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

    def prepare_coords_indices(self, coords_input=None, coords_output=None, indices_sample=None, input_dists=None):

        if indices_sample is not None and isinstance(indices_sample, dict):
            indices_layers = dict(zip(
                self.model.global_levels.tolist(),
                [self.model.get_global_indices_local(indices_sample['sample'],
                                               indices_sample['sample_level'],
                                               global_level)
                                               for global_level in self.model.global_levels]))
            indices_sample['indices_layers'] = indices_layers

        if input_dists is not None and input_dists.numel()==0:
            input_dists = None


        return indices_sample, coords_input, coords_output, input_dists

    def prepare_batch(self, x, coords_input=None, coords_output=None, indices_sample=None, mask=None, emb=None, input_dists=None):
        b, nt = x.shape[:2]
        x = x.view(b * nt, *x.shape[2:])
        if mask is not None:
            mask = mask.view(b * nt, *mask.shape[2:])
        if coords_input is not None:
            coords_input = coords_input.view(b * nt, *coords_input.shape[2:])
        if coords_output is not None:
            coords_output = coords_output.view(b * nt, *coords_output.shape[2:])
        if input_dists is not None:
            input_dists = input_dists.view(b * nt, *input_dists.shape[2:])
        if indices_sample is not None and isinstance(indices_sample, dict):
            for key, value in indices_sample.items():
                indices_sample[key] = value.view(b * nt, *value.shape[2:]) if torch.is_tensor(value) else value
        if emb is not None:
            for key, value in emb.items():
                emb[key] = value.view(b * nt, *value.shape[2:])
            emb['args'] = {}

        return x, coords_input, coords_output, indices_sample, mask, emb, input_dists