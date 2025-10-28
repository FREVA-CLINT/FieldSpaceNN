import os
from typing import Tuple, Dict, Optional, List
import lightning.pytorch as pl
import torch
import math
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from pytorch_lightning.utilities import rank_zero_only

from ...utils.visualization import plot_images


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_iters, iter_start=0):
        self.max_num_iters = max_iters
        self.iter_start = iter_start

        # Fetch per-group warmup and zero_iters from optimizer.param_groups
        self.warmups = [group.get("warmup", 1) for group in optimizer.param_groups]
        self.zero_iters = [group.get("zero_iters", 0) for group in optimizer.param_groups]

        super().__init__(optimizer)

    def get_lr(self):
        factor = self.get_lr_factors(self.last_epoch)
        return [base_lr * f for base_lr, f in zip(self.base_lrs, factor)]

    def get_lr_factors(self, epoch):
        epoch += self.iter_start
        lr_factors = [
            0.5 * (1 + math.cos(math.pi * epoch / self.max_num_iters))
            for _ in self.optimizer.param_groups
        ]

        for i in range(len(lr_factors)):
            if epoch < self.zero_iters[i]:
                lr_factors[i] = 0.0
            elif epoch <= self.warmups[i] and self.warmups[i]>0:
                lr_factors[i] *= epoch / (self.warmups[i])
            elif epoch <= self.warmups[i] and self.warmups[i]==0:
                lr_factors[i] *= epoch

        return lr_factors

class LightningCNN(pl.LightningModule):
    """
    A PyTorch Lightning Module for training and validating a Variational Autoencoder (VAE) model.
    Includes Exponential Moving Average (EMA) for stable parameter updates and a learning rate scheduler with
    cosine annealing and warm-up restarts.
    """

    def __init__(self, model, lr: float, lr_warmup: Optional[int] = None, loss: Optional[_Loss] = None):
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
        self.weight_decay = 0
        self.model = model  # Main VAE model
        self.lr = lr  # Learning rate for optimizer
        self.lr_warmup = lr_warmup  # Warm-up steps if applicable
        self.loss = loss or torch.nn.MSELoss()  # Use MSE if no custom loss is provided
        self.save_hyperparameters(ignore=['model'])  # Save hyperparameters, excluding model



    def forward(self, source_data: Tensor, embeddings: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through the VAE model.

        :param target_data: Ground truth data tensor.
        :param embeddings: Embedding tensor, optional.
        :param mask_data: Mask data tensor, optional.
        :param source_data: Conditioning data tensor, optional.
        :param coords: Coordinates data tensor, optional.

        :return: Tuple containing model output tensor and posterior distribution tensor.
        """
        return self.model(source_data, embeddings)
    

    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        Executes a single training step by calculating reconstruction and KL losses, then logging the results.

        :param batch: A tuple containing input tensors for model training.
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

    def validation_step(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor], batch_idx: int) -> None:
        """
        Executes a single validation step, calculating and logging validation losses, and optionally plotting reconstructions.

        :param batch: A tuple containing input tensors for validation.
        :param batch_idx: The index of the current batch.
        """

        source_data, target_data, source_coords, target_coords, mask_data, emb = batch

        output = self(source_data, emb)

        # Compute reconstruction loss
        loss = self.loss(target_data, output)
    
        # Plot reconstruction samples on the first batch
        if batch_idx == 0 and rank_zero_only.rank == 0:
            self.log_tensor_plot(target_data, source_data, output, target_coords, source_coords, f"tensor_plot_{self.current_epoch}")

        # Log validation losses
        self.log("val_loss", loss.mean(), sync_dist=True)
        self.log_dict({
            'val/total_loss': loss.mean()
        }, sync_dist=True)

    def predict_step(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor], batch_idx: int) -> None:
        """
        Executes a single validation step, calculating and logging validation losses, and optionally plotting reconstructions.

        :param batch: A tuple containing input tensors for validation.
        :param batch_idx: The index of the current batch.
        """
        source_data, target_data, source_coords, target_coords, mask_data, emb = batch

        outputs = self(source_data, emb)

        return {"output": outputs}

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

    def configure_optimizers(self):
        grouped_params = {group_name: [] for group_name in self.lr_groups}
        grouped_params["default"] = []  # fallback group
        seen_params = set()
        
        class_names = []
        def visit_module(module):
            # Match this module by class name
            module_class_name = module.__class__.__name__
            class_names.append(module_class_name)
            matched = False
            for group_name, group_cfg in self.lr_groups.items():
                match_keys = group_cfg.get("matches", [group_name])
                if any(mk in module_class_name for mk in match_keys):
                    matched = True
                    break

            if matched:
                for p in module.parameters():
                    if id(p) not in seen_params and p.requires_grad:
                        grouped_params[group_name].append(p)
                        seen_params.add(id(p))

            # Recurse into submodules (including inside ModuleList/Dict)
            for name, child in module._modules.items():
                if isinstance(child, (nn.ModuleList, nn.ModuleDict)):
                    for sub_child in child.values() if isinstance(child, nn.ModuleDict) else child:
                        visit_module(sub_child)
                elif isinstance(child, nn.Module):
                     visit_module(child)

        # Start recursive traversal from self (i.e. the whole model)
        visit_module(self)

        # Assign leftover parameters to default group
        for p in self.parameters():
            if id(p) not in seen_params and p.requires_grad:
                grouped_params["default"].append(p)
                seen_params.add(id(p))

        param_groups = []
        for group_name, group_cfg in self.lr_groups.items():
            param_groups.append({
                "params": grouped_params[group_name],
                "lr": group_cfg["lr"],
                "name": group_name,
                **{k: v for k, v in group_cfg.items() if k not in {"matches", "lr"}}
            })

        optimizer = torch.optim.Adam(param_groups, weight_decay=self.weight_decay)

        scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            max_iters=self.trainer.max_steps,
            iter_start=0
        )
        
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
