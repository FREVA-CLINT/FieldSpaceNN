import os
from typing import Tuple, Dict

import lightning.pytorch as pl
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from ..mgno_transformer import pl_mgno_base_model

from ...utils.visualization import plot_images
import math

class LightningCNNModel(pl_mgno_base_model.LightningMGNOBaseModel):
    def __init__(self, model, lr_groups, lambda_loss_dict: dict):

        super().__init__(model, lr_groups, lambda_loss_dict, weight_decay=0, noise_std=0, interpolator_settings=None)

        """
        Initializes the LightningDiffusionGenerator with model, diffusion process, and optimizer parameters.

        :param model: The main model for generating diffusion-based images.
        :param gaussian_diffusion: The diffusion process used for training losses.
        :param lr: Learning rate for optimizer.
        :param lr_warmup: Warm-up steps for learning rate. Defaults to None.
        :param ema_rate: Rate for Exponential Moving Average of model parameters. Defaults to 0.999.
        """


 
    def forward(self, x, **kwargs) -> Tuple[Dict[str, Tensor], Tensor]:
        """
        Forward pass through the model for training loss computation.

        :param gt_data: Ground truth data.
        :param diffusion_steps: Steps of the diffusion process.
        :param mask: Mask data. Defaults to None.
        :param cond_data: Conditioning data. Defaults to None.
        :param emb: embedding dictionary

        :return: Dictionary containing loss values for the training step and generated tensor.
        """
        b, n, w, h, nv, nc = x.shape

        x = x.view(b,n,w,h,1,-1)

        x = self.model(x)

        x = x.view(b,n,w,h,-1,1)

        return x
    
    def log_tensor_plot(self, in_tensor: torch.Tensor, rec_tensor: torch.Tensor, gt_tensor: torch.Tensor,
                        in_coords: torch.Tensor, gt_coords: torch.Tensor, mask, indices_dict, plot_name: str, emb, **kwargs):
        
        """
        Logs a plot of ground truth and reconstructed tensor images for validation.

        :param gt_tensor: Ground truth tensor.
        :param rec_tensor: Reconstructed tensor.
        :param plot_name: Name for the plot to be saved.
        """
        save_dir = os.path.join(self.trainer.logger.save_dir, "validation_images")
        os.makedirs(save_dir, exist_ok=True)
        plot_images(gt_tensor, in_tensor, rec_tensor, f"{plot_name}", save_dir, gt_coords, in_coords)

        for c in range(gt_tensor.shape[1]):
            try:
                filename = os.path.join(save_dir, f"{plot_name}_{c}.png")
                self.logger.log_image(f"plots/{plot_name}", [filename])
            except Exception:
                pass