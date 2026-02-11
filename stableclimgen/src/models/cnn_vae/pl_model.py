import os
from typing import Any, Dict, List, Optional, Tuple
import lightning.pytorch as pl
import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.optim import AdamW, Optimizer

from torch.optim.lr_scheduler import LRScheduler
from pytorch_lightning.utilities import rank_zero_only

from ...utils.schedulers import CosineWarmupScheduler

from .model import CNN_VAE


class LightningVAE(pl.LightningModule):
    """
    A PyTorch Lightning Module for training and validating a Variational Autoencoder (VAE) model.
    Includes Exponential Moving Average (EMA) for stable parameter updates and a learning rate scheduler with
    cosine annealing and warm-up restarts.
    """

    def __init__(
        self,
        model: CNN_VAE,
        lr: float,
        lr_warmup: Optional[int] = None,
        ema_rate: float = 0.999,
        loss: Optional[_Loss] = None,
        kl_weight: float = 1e-6,
        mode: str = "encode_decode",
    ) -> None:
        """
        Initializes the LightningVAE with model, optimizer parameters, and optional EMA and warm-up configurations.

        :param model: The VAE model to be trained.
        :param lr: Initial learning rate for the optimizer.
        :param lr_warmup: Optional number of warm-up steps for the learning rate scheduler.
        :param ema_rate: Decay rate for EMA of model parameters, default is 0.999.
        :param loss: Loss function for reconstruction; defaults to Mean Squared Error (MSE).
        :param kl_weight: Weight for the Kullback-Leibler (KL) divergence loss term.
        :param mode: Inference mode for prediction ("encode_decode", "encode", "decode").
        :return: None.
        """
        super().__init__()
        self.model: CNN_VAE = model  # Main VAE model
        self.ema_model: torch.optim.swa_utils.AveragedModel = torch.optim.swa_utils.AveragedModel(
            self.model,
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(ema_rate)
        )
        self.ema_model.requires_grad_(False)  # Prevents gradient updates for EMA model
        self.lr: float = lr  # Learning rate for optimizer
        self.lr_warmup: Optional[int] = lr_warmup  # Warm-up steps if applicable
        self.loss: _Loss = loss or torch.nn.MSELoss()  # Use MSE if no custom loss is provided
        self.kl_weight: float = kl_weight  # KL divergence weighting factor
        self.mode: str = mode
        self.save_hyperparameters(ignore=['model'])  # Save hyperparameters, excluding model

    def on_before_zero_grad(self, optimizer: Optimizer) -> None:
        """
        Updates EMA model parameters before gradients are zeroed.

        :param optimizer: The optimizer instance used in training.
        :return: None.
        """
        self.ema_model.update_parameters(self.model)

    def on_train_end(self) -> None:
        """
        Finalizes EMA model updates by recalculating BatchNorm statistics at the end of training.
        """
        torch.optim.swa_utils.update_bn(self.trainer.train_dataloader, self.ema_model)

    def forward(
        self,
        target_data: Tensor,
        embeddings: Optional[Tensor] = None,
        mask_data: Optional[Tensor] = None,
        source_data: Optional[Tensor] = None,
        coords: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Any]:
        """
        Forward pass through the VAE model.

        :param target_data: Ground truth data tensor of shape ``(b, v, t, n, d, f)`` or
            a CNN-friendly view such as ``(b, c, h, w)`` depending on the model.
        :param embeddings: Optional embedding tensor aligned with ``target_data``.
        :param mask_data: Optional mask tensor aligned with ``target_data``.
        :param source_data: Optional conditioning data tensor.
        :param coords: Optional coordinates tensor.
        :return: Tuple containing model output tensor and posterior distribution.
        """
        return self.model(target_data, embeddings, mask_data, source_data, coords)

    def training_step(
        self,
        batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Any],
        batch_idx: int,
    ) -> Tensor:
        """
        Executes a single training step by calculating reconstruction and KL losses, then logging the results.

        :param batch: Tuple ``(source_data, target_data, source_coords, target_coords, mask_data, emb)``
            where tensors follow the base shape ``(b, v, t, n, d, f)`` or a CNN-friendly view.
        :param batch_idx: The index of the current batch.

        :return: Total calculated loss for the current batch.
        """
        source_data, target_data, source_coords, target_coords, mask_data, emb = batch
        reconstructions, posterior = self(target_data, emb, mask_data)

        # Compute reconstruction loss
        rec_loss = self.loss(target_data, reconstructions)
        # Compute KL divergence loss
        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # Calculate total loss
        loss = rec_loss + self.kl_weight * kl_loss

        # Log individual and total losses
        self.log("train_loss", loss.mean(), prog_bar=True, sync_dist=True)
        self.log_dict({
            'train/kl_loss': kl_loss.mean(),
            'train/rec_loss': rec_loss.mean(),
            'train/total_loss': loss.mean()
        }, prog_bar=True, sync_dist=True)
        return loss.mean()

    def validation_step(
        self,
        batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Any],
        batch_idx: int,
    ) -> None:
        """
        Executes a single validation step, calculating and logging validation losses, and optionally plotting reconstructions.

        :param batch: Tuple ``(source_data, target_data, source_coords, target_coords, mask_data, emb)``
            where tensors follow the base shape ``(b, v, t, n, d, f)`` or a CNN-friendly view.
        :param batch_idx: The index of the current batch.
        """
        source_data, target_data, source_coords, target_coords, mask_data, emb = batch
        reconstructions, posterior = self(target_data, emb, mask_data)

        # Calculate reconstruction and KL losses
        rec_loss = self.loss(target_data, reconstructions)
        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # Calculate total validation loss
        loss = rec_loss + self.kl_weight * kl_loss

        # Plot reconstruction samples on the first batch
        if batch_idx == 0 and rank_zero_only.rank == 0:
            self.logger.log_regular_tensor_plot(target_data, source_data, reconstructions, target_coords, source_coords, f"tensor_plot_{self.current_epoch}")

        # Log validation losses
        self.log("val_loss", loss.mean(), sync_dist=True)
        self.log_dict({
            'val/kl_loss': kl_loss.mean(),
            'val/rec_loss': rec_loss.mean(),
            'val/total_loss': loss.mean()
        }, sync_dist=True)

    def predict_step(
        self,
        batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Any],
        batch_idx: int,
    ) -> dict[str, Tensor]:
        """
        Executes a single validation step, calculating and logging validation losses, and optionally plotting reconstructions.

        :param batch: Tuple ``(source_data, target_data, source_coords, target_coords, mask_data, emb)``
            where tensors follow the base shape ``(b, v, t, n, d, f)`` or a CNN-friendly view.
        :param batch_idx: The index of the current batch.
        :return: Dictionary with output predictions and ground truth tensors.
        """
        source_data, target_data, source_coords, target_coords, mask_data, emb = batch

        if self.mode == "encode_decode":
            outputs, posterior = self(target_data, emb, mask_data)
        elif self.mode == "encode":
            outputs = self.model.encode(target_data).sample()
        elif self.mode == "decode":
            outputs = self.model.decode(source_data)
        return {"output": outputs, "gt": target_data}

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[Dict[str, LRScheduler]]]:
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
