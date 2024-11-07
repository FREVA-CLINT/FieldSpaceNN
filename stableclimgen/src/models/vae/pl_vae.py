import os
from typing import Tuple, Dict, Optional, List
import lightning.pytorch as pl
import torch
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from .model import VAE
from ...utils.visualization import plot_images


class LightningVAE(pl.LightningModule):
    """
    A PyTorch Lightning Module for training and validating a Variational Autoencoder (VAE) model with Exponential Moving Average (EMA)
    and cosine annealing warm-up restarts for the learning rate scheduler.
    """

    def __init__(self, model: VAE, lr: float, lr_warmup: Optional[int] = None, ema_rate: float = 0.999,
                 loss: Optional[_Loss] = None, kl_weight: float = 0.000001):
        """
        Initializes the LightningVAE with model, optimizer parameters, and optional EMA and warm-up configurations.

        :param model: The VAE model to be trained.
        :param lr: Initial learning rate for the optimizer.
        :param lr_warmup: Optional number of warm-up steps for the learning rate scheduler.
        :param ema_rate: Decay rate for EMA of model parameters, default is 0.999.
        :param loss: Loss function for reconstruction; defaults to Mean Squared Error (MSE).
        :param kl_weight: Weight for the Kullback-Leibler (KL) divergence loss term.
        """
        super().__init__()
        self.model: VAE = model  # The main VAE model instance
        self.ema_model = torch.optim.swa_utils.AveragedModel(
            self.model,
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(ema_rate)
        )
        self.ema_model.requires_grad_(False)  # Prevents gradient updates for EMA model
        self.lr = lr  # Learning rate for optimizer
        self.lr_warmup = lr_warmup  # Warm-up steps for the scheduler if provided
        self.loss = loss or torch.nn.MSELoss()  # Default loss is MSE if no custom loss is provided
        self.kl_weight = kl_weight  # Weight for KL divergence in the loss calculation
        self.save_hyperparameters(ignore=['model'])

    def on_before_zero_grad(self, optimizer: Optimizer) -> None:
        """
        Updates the EMA model parameters before zeroing gradients to maintain moving average.

        :param optimizer: The optimizer instance used in training.
        """
        self.ema_model.update_parameters(self.model)

    def on_train_end(self) -> None:
        """
        Finalizes EMA model updates by recalculating BatchNorm statistics at the end of training.
        """
        torch.optim.swa_utils.update_bn(self.trainer.train_dataloader, self.ema_model)

    def forward(self, gt_data: Tensor, embeddings: Optional[Tensor] = None, mask_data: Optional[Tensor] = None,
                cond_data: Optional[Tensor] = None, coords: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through the model, typically for loss calculation.

        :param gt_data: Ground truth data tensor.
        :param embeddings: Embedding tensor used in the diffusion process, optional.
        :param mask_data: Mask data tensor, optional.
        :param cond_data: Conditioning data tensor, optional.
        :param coords: Coordinates data tensor, optional.

        :return: Tuple containing model output tensor and posterior distribution tensor.
        """
        return self.model(gt_data, embeddings, mask_data, cond_data, coords)

    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        Executes a single training step by calculating reconstruction and KL losses, logging the results.

        :param batch: A tuple containing input tensors for model training.
        :param batch_idx: The index of the current batch.

        :return: Total calculated loss for the current batch.
        """
        cond_data, cond_coords, gt_data, gt_coords, mask_data = batch
        reconstructions, posterior = self(gt_data)

        # Compute reconstruction and KL losses
        rec_loss = self.loss(gt_data, reconstructions)
        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # Total loss with KL divergence weighted by kl_weight
        loss = rec_loss + self.kl_weight * kl_loss

        # Log individual losses and total loss
        self.log("train_loss", loss.mean(), prog_bar=True, sync_dist=True)
        loss_dict = {
            'train/kl_loss': kl_loss.mean(),
            'train/rec_loss': rec_loss.mean(),
            'train/total_loss': loss.mean()
        }
        self.log_dict(loss_dict, prog_bar=True, sync_dist=True)
        return loss.mean()

    def validation_step(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor], batch_idx: int) -> None:
        """
        Executes a single validation step, calculates and logs validation losses, and plots images if applicable.

        :param batch: A tuple containing input tensors for validation.
        :param batch_idx: The index of the current batch.
        """
        cond_data, cond_coords, gt_data, gt_coords, mask_data = batch
        reconstructions, posterior = self(gt_data)

        # Compute reconstruction and KL losses for validation
        rec_loss = self.loss(gt_data, reconstructions)
        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # Total validation loss
        loss = rec_loss + self.kl_weight * kl_loss

        # Plot reconstructions if on the first batch
        if batch_idx == 0:
            self.log_tensor_plot(gt_data, reconstructions, f"tensor_plot_{self.current_epoch}")

        # Log individual validation losses and total loss
        self.log("val_loss", loss.mean(), sync_dist=True)
        loss_dict = {
            'val/kl_loss': kl_loss.mean(),
            'val/rec_loss': rec_loss.mean(),
            'val/total_loss': loss.mean()
        }
        self.log_dict(loss_dict, sync_dist=True)

    def log_tensor_plot(self, gt_tensor: Tensor, rec_tensor: Tensor, plot_name: str) -> None:
        """
        Saves and logs a plot of ground truth and reconstructed tensors for visualization.

        :param gt_tensor: Ground truth tensor.
        :param rec_tensor: Reconstructed tensor.
        :param plot_name: Name of the plot to save.
        """
        save_dir = os.path.join(self.trainer.logger.save_dir, "validation_images")
        os.makedirs(save_dir, exist_ok=True)
        plot_images(gt_tensor, rec_tensor, f"{plot_name}", save_dir)

        for c in range(gt_tensor.shape[1]):
            filename = os.path.join(save_dir, f"{plot_name}_{c}.png")
            self.logger.log_image(f"plots/{plot_name}", [filename])

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[Dict[str, LRScheduler]]]:
        """
        Configures the AdamW optimizer and learning rate scheduler for model training.

        :return: A tuple containing optimizer and scheduler configurations.
        """
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=0.0)

        if self.lr_warmup:
            # Calculate total training steps based on data size and training configuration
            dataset = self.trainer.fit_loop._data_source.dataloader()
            dataset_size = len(dataset)
            steps = dataset_size * self.trainer.max_epochs // (
                    self.trainer.accumulate_grad_batches * max(1, self.trainer.num_devices)
            )
            # Scheduler with cosine annealing warm-up restarts
            scheduler = CosineAnnealingWarmupRestarts(
                optimizer=optimizer,
                first_cycle_steps=steps,
                max_lr=self.lr,
                min_lr=1E-6,
                warmup_steps=self.lr_warmup
            )
        else:
            # Constant learning rate without scheduler adjustments
            scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
