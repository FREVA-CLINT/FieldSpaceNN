import os
from typing import Tuple, Dict, Optional, List
import lightning.pytorch as pl
import torch
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from pytorch_lightning.utilities import rank_zero_only

from .model import VAE
from ...utils.visualization import plot_images


class LightningVAE(pl.LightningModule):
    """
    A PyTorch Lightning Module for training and validating a Variational Autoencoder (VAE) model.
    Includes Exponential Moving Average (EMA) for stable parameter updates and a learning rate scheduler with
    cosine annealing and warm-up restarts.
    """

    def __init__(self, model: VAE, lr: float, lr_warmup: Optional[int] = None, ema_rate: float = 0.999,
                 loss: Optional[_Loss] = None, kl_weight: float = 1e-6, mode="encode_decode"):
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
        self.model: VAE = model  # Main VAE model
        self.ema_model = torch.optim.swa_utils.AveragedModel(
            self.model,
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(ema_rate)
        )
        self.ema_model.requires_grad_(False)  # Prevents gradient updates for EMA model
        self.lr = lr  # Learning rate for optimizer
        self.lr_warmup = lr_warmup  # Warm-up steps if applicable
        self.loss = loss or torch.nn.MSELoss()  # Use MSE if no custom loss is provided
        self.kl_weight = kl_weight  # KL divergence weighting factor
        self.mode = mode
        self.save_hyperparameters(ignore=['model'])  # Save hyperparameters, excluding model

    def on_before_zero_grad(self, optimizer: Optimizer) -> None:
        """
        Updates EMA model parameters before gradients are zeroed.

        :param optimizer: The optimizer instance used in training.
        """
        self.ema_model.update_parameters(self.model)

    def on_train_end(self) -> None:
        """
        Finalizes EMA model updates by recalculating BatchNorm statistics at the end of training.
        """
        torch.optim.swa_utils.update_bn(self.trainer.train_dataloader, self.ema_model)

    def forward(self, target_data: Tensor, embeddings: Optional[Tensor] = None, mask_data: Optional[Tensor] = None,
                source_data: Optional[Tensor] = None, coords: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through the VAE model.

        :param target_data: Ground truth data tensor.
        :param embeddings: Embedding tensor, optional.
        :param mask_data: Mask data tensor, optional.
        :param source_data: Conditioning data tensor, optional.
        :param coords: Coordinates data tensor, optional.

        :return: Tuple containing model output tensor and posterior distribution tensor.
        """
        return self.model(target_data, embeddings, mask_data, source_data, coords)

    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        Executes a single training step by calculating reconstruction and KL losses, then logging the results.

        :param batch: A tuple containing input tensors for model training.
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

    def validation_step(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor], batch_idx: int) -> None:
        """
        Executes a single validation step, calculating and logging validation losses, and optionally plotting reconstructions.

        :param batch: A tuple containing input tensors for validation.
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
            self.log_tensor_plot(target_data, source_data, reconstructions, target_coords, source_coords, f"tensor_plot_{self.current_epoch}")

        # Log validation losses
        self.log("val_loss", loss.mean(), sync_dist=True)
        self.log_dict({
            'val/kl_loss': kl_loss.mean(),
            'val/rec_loss': rec_loss.mean(),
            'val/total_loss': loss.mean()
        }, sync_dist=True)

    def predict_step(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor], batch_idx: int) -> None:
        """
        Executes a single validation step, calculating and logging validation losses, and optionally plotting reconstructions.

        :param batch: A tuple containing input tensors for validation.
        :param batch_idx: The index of the current batch.
        """
        source_data, target_data, source_coords, target_coords, mask_data, emb = batch

        if self.mode == "encode_decode":
            outputs, posterior = self(target_data, emb, mask_data)
        elif self.mode == "encode":
            outputs = self.model.encode(target_data).sample()
        elif self.mode == "decode":
            outputs = self.model.decode(source_data)
        return outputs

    def log_tensor_plot(self, gt_tensor: torch.Tensor, in_tensor: torch.Tensor, rec_tensor: torch.Tensor,
                        target_coords: torch.Tensor, in_coords: torch.Tensor, plot_name: str):
        """
        Logs a plot of ground truth and reconstructed tensor images for visualization.

        :param gt_tensor: Ground truth tensor.
        :param in_tensor: Input tensor.
        :param rec_tensor: Reconstructed tensor.
        :param target_coords: Ground truth coordinates tensor.
        :param in_coords: Input coordinates tensor.
        :param plot_name: Name for the plot to be saved.
        """
        save_dir = os.path.join(self.trainer.logger.save_dir, "validation_images")
        os.makedirs(save_dir, exist_ok=True)
        plot_images(gt_tensor, in_tensor, rec_tensor, plot_name, save_dir, target_coords, in_coords)

        # Log images for each channel
        for c in range(gt_tensor.shape[1]):
            filename = os.path.join(save_dir, f"{plot_name}_{c}.png")
            self.logger.log_image(f"plots/{plot_name}", [filename])

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[Dict[str, LRScheduler]]]:
        """
        Configures the optimizer and learning rate scheduler for training.

        :return: A tuple containing the optimizer and scheduler configurations.
        """
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=0.0)

        # Apply cosine annealing with warm-up if specified
        if self.lr_warmup:
            dataset = self.trainer.fit_loop._data_source.dataloader()
            dataset_size = len(dataset)
            steps = dataset_size * self.trainer.max_epochs // (
                    self.trainer.accumulate_grad_batches * max(1, self.trainer.num_devices)
            )
            scheduler = CosineAnnealingWarmupRestarts(
                optimizer=optimizer,
                first_cycle_steps=steps,
                max_lr=self.lr,
                min_lr=1E-6,
                warmup_steps=self.lr_warmup
            )
        else:
            # Use constant learning rate if no warm-up is specified
            scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
