import os
from typing import Any, Dict, List, Optional, Tuple
import lightning.pytorch as pl
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.loss import _Loss
from pytorch_lightning.utilities import rank_zero_only
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler

from .model import CNN

from ...utils.schedulers import CosineWarmupScheduler
from ...utils.visualization import regular_plot

class LightningCNN(pl.LightningModule):
    """
    A PyTorch Lightning Module for training and validating a CNN model.
    Includes a cosine learning rate scheduler with warm-up.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float,
        lr_warmup: Optional[int] = None,
        loss: Optional[_Loss] = None,
    ):
        """
        Initializes the LightningCNN with model, optimizer parameters, and optional warm-up configurations.

        :param model: The CNN model to be trained.
        :param lr: Initial learning rate for the optimizer.
        :param lr_warmup: Optional number of warm-up steps for the learning rate scheduler.
        :param loss: Loss function for reconstruction; defaults to Mean Squared Error (MSE).
        :return: None.
        """
        super().__init__()
        self.weight_decay: float = 0.0
        self.model: CNN = model  # Main CNN model
        self.lr: float = lr  # Learning rate for optimizer
        self.lr_warmup: Optional[int] = lr_warmup  # Warm-up steps if applicable
        self.loss: _Loss = loss or torch.nn.MSELoss()  # Use MSE if no custom loss is provided
        self.save_hyperparameters(ignore=['model'])  # Save hyperparameters, excluding model



    def forward(self, source_data: Tensor, embeddings: Optional[Dict[str, Any]] = None):
        """
        Forward pass through the CNN model.

        :param source_data: Input tensor of shape ``(b, v, t, n, d, f)`` or a CNN-friendly
            view such as ``(b, c, h, w)`` depending on the model configuration.
        :param embeddings: Optional embedding dictionary aligned with ``source_data``.
        :return: Model output tensor matching the target shape.
        """
        return self.model(source_data, embeddings)
    

    def training_step(
        self,
        batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Any],
        batch_idx: int,
    ):
        """
        Executes a single training step by calculating reconstruction and KL losses, then logging the results.

        :param batch: Tuple ``(source_data, target_data, source_coords, target_coords, mask_data, emb)``
            where tensors follow the base shape ``(b, v, t, n, d, f)`` (often flattened
            to a CNN-friendly view such as ``(b, c, h, w)``). ``emb`` contains optional
            embeddings aligned with ``source_data``.
        :param batch_idx: The index of the current batch.

        :return: Total calculated loss for the current batch.
        """
        source_data, target_data, source_coords, target_coords, mask_data, emb = batch

        output = self(source_data, emb)

        # Compute reconstruction loss
        loss = self.loss(target_data, output)
    

        # Log individual and total losses
        self.log("train_loss", loss.mean(), prog_bar=True, sync_dist=True)
        self.log_dict({
            'train/total_loss': loss.mean()
        }, prog_bar=True, sync_dist=True)
        return loss.mean()

    def validation_step(
        self,
        batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Any],
        batch_idx: int,
    ):
        """
        Executes a single validation step, calculating and logging validation losses, and optionally plotting reconstructions.

        :param batch: Tuple ``(source_data, target_data, source_coords, target_coords, mask_data, emb)``
            with tensors in the base shape ``(b, v, t, n, d, f)`` or a CNN-friendly view.
        :param batch_idx: The index of the current batch.
        """

        source_data, target_data, source_coords, target_coords, mask_data, emb = batch

        output = self(source_data, emb)

        # Compute reconstruction loss
        loss = self.loss(target_data, output)
    
        # Plot reconstruction samples on the first batch
        if batch_idx == 0 and rank_zero_only.rank == 0:
            self.logger.log_regular_tensor_plot(target_data, source_data, output, target_coords, source_coords, f"tensor_plot_{self.current_epoch}")

        # Log validation losses
        self.log("val_loss", loss.mean(), sync_dist=True)
        self.log_dict({
            'val/total_loss': loss.mean()
        }, sync_dist=True)

    def predict_step(
        self,
        batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Any],
        batch_idx: int,
    ):
        """
        Executes a single validation step, calculating and logging validation losses, and optionally plotting reconstructions.

        :param batch: Tuple ``(source_data, target_data, source_coords, target_coords, mask_data, emb)``
            with tensors in the base shape ``(b, v, t, n, d, f)`` or a CNN-friendly view.
        :param batch_idx: The index of the current batch.
        :return: Dictionary with the model output tensor.
        """
        source_data, target_data, source_coords, target_coords, mask_data, emb = batch

        outputs = self(source_data, emb)

        return {"output": outputs}

    def log_tensor_plot(
        self,
        gt_tensor: torch.Tensor,
        in_tensor: torch.Tensor,
        rec_tensor: torch.Tensor,
        target_coords: torch.Tensor,
        in_coords: torch.Tensor,
        plot_name: str,
    ):
        """
        Logs a plot of ground truth and reconstructed tensor images for visualization.

        :param gt_tensor: Ground truth tensor of shape ``(b, v, t, n, d, f)`` or a
            CNN-friendly view such as ``(b, c, h, w)``.
        :param in_tensor: Input tensor aligned with ``gt_tensor``.
        :param rec_tensor: Reconstructed tensor aligned with ``gt_tensor``.
        :param target_coords: Ground truth coordinates tensor.
        :param in_coords: Input coordinates tensor.
        :param plot_name: Name for the plot to be saved.
        :return: None.
        """
        save_dir = os.path.join(self.trainer.logger.save_dir, "validation_images")
        os.makedirs(save_dir, exist_ok=True)
        regular_plot(gt_tensor, in_tensor, rec_tensor, plot_name, save_dir, target_coords, in_coords)

        # Log images for each channel
        for c in range(gt_tensor.shape[1]):
            filename = os.path.join(save_dir, f"{plot_name}_{c}.png")
            self.logger.log_image(f"plots/{plot_name}", [filename])

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler for training.

        :return: A tuple containing the optimizer and scheduler configurations.
        """
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=0.0)

        scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            max_iters=self.trainer.max_steps,
            iter_start=0
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
