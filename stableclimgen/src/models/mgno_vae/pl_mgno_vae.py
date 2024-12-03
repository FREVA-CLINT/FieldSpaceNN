import os
import math
import torch.nn as nn

import lightning.pytorch as pl
import torch

from torch.optim import AdamW

from ...utils.visualization import scatter_plot

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


class MSE_loss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss_fcn = torch.nn.MSELoss() 

    def forward(self, output, target):
        loss = self.loss_fcn(output, target.view(output.shape))
        return loss


class Lightning_MGNO_VAE(pl.LightningModule):
    def __init__(self, model, lr, lr_warmup=None):
        super().__init__()
        # maybe create multi_grid structure here?
        self.model = model

        self.lr = lr
        self.lr_warmup = lr_warmup
        self.save_hyperparameters(ignore=['model'])
        self.loss = MSE_loss()

    def forward(self, x, coords_input, coords_output, indices_sample=None, mask=None, emb=None):
        x: torch.tensor = self.model(x, coords_input=coords_input, coords_output=coords_output, indices_sample=indices_sample, mask=mask, emb=emb)
        return x

    def training_step(self, batch, batch_idx):
        source, target, coords_input, coords_output, indices, mask, emb = batch
        output = self(source, coords_input=coords_input, coords_output=coords_output, indices_sample=indices, mask=mask, emb=emb)
        loss = self.loss(output, target)
        self.log_dict({"train/loss": loss.item()}, prog_bar=True, sync_dist=True)
        return loss


    def validation_step(self, batch, batch_idx):
        source, target, coords_input, coords_output, indices, mask, emb = batch
        output = self(source, coords_input=coords_input, coords_output=coords_output, indices_sample=indices, mask=mask, emb=emb)
        loss = self.loss(output, target)
        self.log_dict({"loss": loss.item()}, sync_dist=True)

        if batch_idx == 0:
            self.log_tensor_plot(source, output, target, coords_input, coords_output, mask, indices,f"tensor_plot_{self.current_epoch}", emb)

        return loss

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
                if indices_dict is not None and isinstance(indices_dict, dict):
                    indices = self.model.get_global_indices_local(indices_dict['sample'], 
                                                                                indices_dict['sample_level'], 
                                                                                0)
                    coords_input = coords_output = self.model.cell_coords_global[indices]
                else:
                    coords_input = coords_output = self.model.cell_coords_global.unsqueeze(dim=0)
            
            if coords_input.shape[0]==1:
                coords_input_plot = coords_input[0]
                coords_output_plot = coords_output[0]
            else:
                coords_input_plot = coords_input[sample]
                coords_output_plot = coords_output[sample]

            scatter_plot(input[sample,:,:,k], output[sample,:,k], gt[sample,:,:,k], coords_input_plot, coords_output_plot, mask_p, save_path=save_path)
            self.logger.log_image(f"plots/{plot_name_var}", [save_path])



    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=0.0)
        scheduler = CosineWarmupScheduler(
                        optimizer=optimizer,
                        warmup=self.lr_warmup,
                        max_iters=self.trainer.max_steps,
                        iter_start=0)
        
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
