import os
import math
import torch.nn as nn
import xarray as xr

import lightning.pytorch as pl
import torch
from PIL import UnidentifiedImageError
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from utils.visualization import scatter_plot

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters, iter_start=0):
        self.warmup = warmup
        self.max_num_iters = max_iters
        self.iter_start = iter_start
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        epoch += self.iter_start
        lr_factor = 0.5 * (1 + math.cos(math.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor

def dict_to_device(d, device):
    for key, value in d.items():
        d[key] = value.to(device)
    return d

def data_to_device(d, device):
    if isinstance(d, dict):
        return dict_to_device(d, device)
    elif torch.is_tensor(d):
        return d.to(device)
    else:
        return None


class MSE_loss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss_fcn = torch.nn.MSELoss() 

    def forward(self, output, target):
        loss = 0
        idx=0
       # for target_var in target.values():
        loss = self.loss_fcn(output, target.view(output.shape))
       #     idx += 1
        return loss


class LightningICONTransformer(pl.LightningModule):
    def __init__(self, model, lr, lr_warmup=None):
        super().__init__()
        # maybe create multi_grid structure here?
        self.model = model

        self.lr = lr
        self.lr_warmup = lr_warmup
        self.save_hyperparameters(ignore=['model'])
        self.loss = MSE_loss()

    def forward(self, x, coords_input=None, coords_output=None, sampled_indices_batch_dict=None, drop_mask=None):
        x: torch.tensor = self.model(x, coords_input=coords_input, coords_output=coords_output, sampled_indices_batch_dict=sampled_indices_batch_dict, drop_mask=drop_mask)
        return x

    def training_step(self, batch, batch_idx):
        #batch = [data_to_device(x, self.trainer.accelerator) for x in batch]
        source, target, indices, drop_mask, coords_input, coords_output = batch
        output = self(source, coords_input=coords_input, coords_output=coords_output, sampled_indices_batch_dict=indices, drop_mask=drop_mask)
        loss = self.loss(output, target)
        self.log_dict({"train/loss": loss.item()}, prog_bar=True, sync_dist=True)
        return loss


    def validation_step(self, batch, batch_idx):
        source, target, indices, drop_mask, coords_input, coords_output = batch
        output = self(source, coords_input=coords_input, coords_output=coords_output, sampled_indices_batch_dict=indices, drop_mask=drop_mask)
        loss = self.loss(output, target)
        self.log_dict({"loss": loss.item()}, sync_dist=True)

        if batch_idx == 0:
            self.log_tensor_plot(source, output, target, coords_input, coords_output, drop_mask, indices,f"tensor_plot_{self.current_epoch}")

        return loss

    def log_tensor_plot(self, input, output, gt, coords_input, coords_output, mask, indices_dict, plot_name):

        save_dir = os.path.join(self.trainer.logger.save_dir, "validation_images")
        os.makedirs(save_dir, exist_ok=True)  
        
        sample = 0
        for k in range(input.shape[-2]):
            var_idx = indices_dict['variables'][sample, k]
            plot_name_var = f"{plot_name}_{var_idx}"
            save_path = os.path.join(save_dir, f"{plot_name_var}.png")

            if mask is not None:
                mask_p = mask[sample,:,:,k]
            else:
                mask_p = None
            
            if coords_input.numel()==0:
                coords_input = coords_output = self.model.cell_coords_global[:,indices_dict['local_cell']]

            scatter_plot(input[sample,:,:,k], output[sample,:,k], gt[sample,:,:,k], coords_input[:,sample], coords_output[:,sample], mask_p, save_path=save_path)
            self.logger.log_image(f"plots/{plot_name_var}", [save_path])



    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=0.0)
        scheduler = CosineWarmupScheduler(
                        optimizer=optimizer,
                        warmup=self.lr_warmup,
                        max_iters=self.trainer.max_steps,
                        iter_start=0)
        
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
