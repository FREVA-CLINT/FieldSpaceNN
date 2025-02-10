import os
import math
import torch.nn as nn

import lightning.pytorch as pl
import torch

from torch.optim import AdamW

from ...utils.visualization import scatter_plot
from ...modules.icon_grids.grid_layer import GridLayer

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
    def __init__(self):
        super().__init__()

        self.loss_fcn = torch.nn.MSELoss() 

    def forward(self, output, target):
        loss = self.loss_fcn(output, target.view(output.shape))
        return loss
   
class Grad_loss(nn.Module):
    def __init__(self, grid_layer):
        super().__init__()
        self.grid_layer: GridLayer = grid_layer
        self.loss_fcn = torch.nn.KLDivLoss(log_target=True, reduction='batchmean') 

    def forward(self, output, target, indices_sample=None):
        indices_in  = indices_sample["indices_layers"][int(self.grid_layer.global_level)] if indices_sample is not None else None
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

    def forward(self, output, indices_sample=None):
        indices_in  = indices_sample["indices_layers"][int(self.grid_layer.global_level)] if indices_sample is not None else None
        output_nh, _ = self.grid_layer.get_nh(output, indices_in, indices_sample)
        
        loss = ((output_nh[:,:,[0]] - output_nh[:,:,1:])).abs().mean().sqrt()
        return loss


class LightningMGNOTransformer(pl.LightningModule):
    def __init__(self, model, lr_groups, grad_loss_weight=0., nh_loss_weight=0.):
        super().__init__()
        # maybe create multi_grid structure here?
        self.model = model

        self.lr_groups=lr_groups
        self.save_hyperparameters(ignore=['model'])
        self.loss = MSE_loss()
        self.grad_loss_weight = grad_loss_weight
        self.nh_loss_weight = nh_loss_weight
        self.grad_loss = Grad_loss(model.grid_layer_0)
        self.nh_loss = NH_loss(model.grid_layer_0)

    def forward(self, x, coords_input, coords_output, indices_sample=None, mask=None, emb=None):
        x: torch.tensor = self.model(x, coords_input=coords_input, coords_output=coords_output, indices_sample=indices_sample, mask=mask, emb=emb)
        return x

    def training_step(self, batch, batch_idx):
        source, target, coords_input, coords_output, indices, mask, emb = batch
        output = self(source, coords_input=coords_input, coords_output=coords_output, indices_sample=indices, mask=mask, emb=emb)
        
        loss = self.loss(output, target)
        self.log_dict({"train/mse_loss": loss.item()})

        if self.grad_loss_weight>0:
            grad_loss = self.grad_loss(output, target, indices_sample=indices)
            loss = loss + self.grad_loss_weight*grad_loss
            self.log_dict({"train/grad_loss": grad_loss.item()})

        if self.nh_loss_weight>0:
            nh_tv_loss = self.nh_loss(output, indices_sample=indices)
            loss = loss + self.nh_loss_weight*nh_tv_loss
            self.log_dict({"train/nh_tv_loss": nh_tv_loss.item()})

        self.log_dict({"train/loss": loss.item()}, prog_bar=True)
        self.log_dict(self.get_no_params_dict(), logger=True)
        
        return loss


    def validation_step(self, batch, batch_idx):
        source, target, coords_input, coords_output, indices, mask, emb = batch
        output = self(source, coords_input=coords_input, coords_output=coords_output, indices_sample=indices, mask=mask, emb=emb)
        loss = self.loss(output, target)
        self.log_dict({"validate/mse_loss": loss.item()})

        if self.grad_loss_weight>0:
            grad_loss = self.grad_loss(output, target, indices_sample=indices)
            loss = loss + self.grad_loss_weight*grad_loss
            self.log_dict({"validate/grad_loss": grad_loss.item()})

        if self.nh_loss_weight>0:
            nh_tv_loss = self.nh_loss(output, indices_sample=indices)
            loss = loss + self.nh_loss_weight*nh_tv_loss
            self.log_dict({"validate/nh_tv_loss": nh_tv_loss.item()})

        self.log_dict({"validate/loss": loss.item()}, prog_bar=True)
        self.log_dict(self.get_no_params_dict(), logger=True)

        if batch_idx == 0:
            self.log_tensor_plot(source, output, target, coords_input, coords_output, mask, indices,f"tensor_plot_{self.current_epoch}", emb)
            

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

    def log_tensor_plot(self, input, output, gt, coords_input, coords_output, mask, indices_dict, plot_name, emb):

        save_dir = os.path.join(self.trainer.logger.save_dir, "validation_images")
        os.makedirs(save_dir, exist_ok=True)  
        
        sample = 0
        for k in range(input.shape[-2]):
            var_idx = emb['VariableEmbedder'][sample, k]
            plot_name_var = f"{plot_name}_{var_idx}"
            save_path = os.path.join(save_dir, f"{plot_name_var}.png")

            if mask is not None:
                mask_p = mask[sample,:,:,k]
            else:
                mask_p = None
            
            if coords_input.numel()==0:
                coords_input = self.get_coords_from_model(indices_dict)
                
            if coords_output.numel()==0:
                coords_output = self.get_coords_from_model(indices_dict)

            if coords_input.shape[0]==1:
                coords_input_plot = coords_input[0]
                coords_output_plot = coords_output[0]
            else:
                coords_input_plot = coords_input[sample]
                coords_output_plot = coords_output[sample]

            scatter_plot(input[sample,:,:,k], output[sample,:,k], gt[sample,:,:,k], coords_input_plot, coords_output_plot, mask_p, save_path=save_path)
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
            elif "att_block" in n:
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


        optimizer = AdamW(param_groups)
        scheduler = CosineWarmupScheduler(
                        optimizer=optimizer,
                        lr_groups=self.lr_groups,
                        max_iters=self.trainer.max_steps,
                        iter_start=0)
        
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
