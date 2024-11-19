import os
from typing import Tuple, Dict

import lightning.pytorch as pl
import torch
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .model import DiffusionGenerator
from .gaussian_diffusion import GaussianDiffusion
from ...utils.visualization import plot_images


class LightningDiffusionGenerator(pl.LightningModule):
    """
    A PyTorch Lightning Module for training and validating a Diffusion Generator model with Exponential Moving Average (EMA)
    and cosine annealing warm-up restarts for the learning rate scheduler.
    """

    def __init__(self, model: DiffusionGenerator, gaussian_diffusion: GaussianDiffusion, lr: float, lr_warmup: int = None,
                 ema_rate: float = 0.999):
        """
        Initializes the LightningDiffusionGenerator with model, diffusion process, and optimizer parameters.

        :param model: The main model for generating diffusion-based images.
        :param gaussian_diffusion: The diffusion process used for training losses.
        :param lr: Learning rate for optimizer.
        :param lr_warmup: Warm-up steps for learning rate. Defaults to None.
        :param ema_rate: Rate for Exponential Moving Average of model parameters. Defaults to 0.999.
        """
        super().__init__()
        self.model: DiffusionGenerator = model
        self.ema_model = torch.optim.swa_utils.AveragedModel(
            self.model,
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(ema_rate)
        )
        self.ema_model.requires_grad_(False)  # Ensure no gradient updates for EMA model
        self.gaussian_diffusion = gaussian_diffusion
        self.lr = lr
        self.lr_warmup = lr_warmup
        self.save_hyperparameters(ignore=['model'])

    def on_before_zero_grad(self, optimizer):
        """
        Updates the EMA model parameters before zeroing gradients.

        :param optimizer: The optimizer instance being used for training.
        """
        self.ema_model.update_parameters(self.model)

    def on_train_end(self):
        """
        Finalizes the EMA model with updated BatchNorm statistics at the end of training.
        """
        torch.optim.swa_utils.update_bn(self.trainer.train_dataloader, self.ema_model)

    def forward(self, gt_data: torch.Tensor, diffusion_steps: torch.Tensor, mask_data: torch.Tensor = None,
                cond_data: torch.Tensor = None, coords: torch.Tensor = None) -> Tuple[Dict[str, Tensor], Tensor]:
        """
        Forward pass through the model for training loss computation.

        :param gt_data: Ground truth data.
        :param diffusion_steps: Steps of the diffusion process.
        :param mask_data: Mask data. Defaults to None.
        :param cond_data: Conditioning data. Defaults to None.
        :param coords: Coordinates data. Defaults to None.

        :return: Dictionary containing loss values for the training step and generated tensor.
        """
        return self.gaussian_diffusion.training_losses(self.model, gt_data, diffusion_steps, mask_data, cond_data,
                                                       coords)

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        """
        Performs a single training step, calculates losses, and logs them.

        :param batch: Batch of input data.
        :param batch_idx: Index of the current batch.

        :return: Calculated loss for the current batch.
        """
        cond_data, cond_coords, gt_data, gt_coords, mask_data = batch
        diffusion_steps, weights = self.gaussian_diffusion.get_diffusion_steps(gt_data.shape[0], gt_data.device)
        l_dict, _ = self(gt_data, diffusion_steps, mask_data, cond_data, gt_coords)
        loss = (l_dict["loss"] * weights).mean()

        loss_dict = {f'train/{k}': v.mean() for k, v in l_dict.items()}
        self.log_dict(loss_dict, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Tuple, batch_idx: int):
        """
        Performs a single validation step, calculates and logs losses, and optionally plots validation images.

        :param batch: Batch of input data.
        :param batch_idx: Index of the current batch.
        """
        cond_data, cond_coords, gt_data, gt_coords, mask_data = batch
        loss_dict = {}
        loss = []

        # Iterate over batch items and compute validation loss for each
        for i in range(gt_data.shape[0]):
            diff_steps = self.gaussian_diffusion.diffusion_steps
            t = torch.tensor([(diff_steps // 10) * x for x in range(10)]).to(gt_data.device)
            cond = torch.stack(10 * [cond_data[i]])
            l_dict, output = self(torch.stack(10 * [gt_data[i]]), t, torch.stack(10 * [mask_data[i]]), cond, torch.stack(10 * [gt_coords[i]]))

            for k, v in l_dict.items():
                for ti in range(t.shape[0]):
                    loss_dict[f'val/step_{t[ti].item()}_{k}'] = v[ti]
                loss.append(v.mean())

            if batch_idx == 0 and i == 0:
                self.log_tensor_plot(torch.stack(10 * [gt_data[i]]), torch.stack(10 * [cond_data[i]]),
                                     output, gt_coords, cond_coords, f"tensor_plot_{self.current_epoch}")

        self.log_dict(loss_dict, sync_dist=True)
        self.log('val_loss', torch.stack(loss).mean(), sync_dist=True)
        return self.log_dict

    def log_tensor_plot(self, gt_tensor: torch.Tensor, in_tensor: torch.Tensor, rec_tensor: torch.Tensor,
                        gt_coords: torch.Tensor, in_coords: torch.Tensor, plot_name: str):
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
            filename = os.path.join(save_dir, f"{plot_name}_{c}.png")
            self.logger.log_image(f"plots/{plot_name}", [filename])

    def configure_optimizers(self) -> Tuple:
        """
        Configures the optimizer and learning rate scheduler.

        :return: A tuple containing the optimizer and scheduler configurations.
        """
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=0.0)

        if self.lr_warmup:
            dataset = self.trainer.fit_loop._data_source.dataloader()
            dataset_size = len(dataset)
            steps = dataset_size * self.trainer.max_epochs // (
                        self.trainer.accumulate_grad_batches * max(1, self.trainer.num_devices))
            scheduler = CosineAnnealingWarmupRestarts(
                optimizer=optimizer,
                first_cycle_steps=steps,
                max_lr=self.lr,
                min_lr=1E-6,
                warmup_steps=self.lr_warmup
            )
        else:
            scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
