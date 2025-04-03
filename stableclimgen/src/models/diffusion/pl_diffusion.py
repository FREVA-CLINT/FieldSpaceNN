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
from .sampler import DDPMSampler, DDIMSampler
from ..mgno_transformer.pl_probabilistic import LightningProbabilisticModel
from ...utils.visualization import plot_images


class LightningDiffusionGenerator(LightningProbabilisticModel):
    """
    A PyTorch Lightning Module for training and validating a Diffusion Generator model with Exponential Moving Average (EMA)
    and cosine annealing warm-up restarts for the learning rate scheduler.
    """

    def __init__(self, model: DiffusionGenerator, gaussian_diffusion: GaussianDiffusion, lr: float, lr_warmup: int = None,
                 ema_rate: float = 0.999, sampler="ddpm", n_samples=1, max_batchsize=-1):
        """
        Initializes the LightningDiffusionGenerator with model, diffusion process, and optimizer parameters.

        :param model: The main model for generating diffusion-based images.
        :param gaussian_diffusion: The diffusion process used for training losses.
        :param lr: Learning rate for optimizer.
        :param lr_warmup: Warm-up steps for learning rate. Defaults to None.
        :param ema_rate: Rate for Exponential Moving Average of model parameters. Defaults to 0.999.
        """
        super().__init__()
        LightningProbabilisticModel.__init__(self, n_samples=1, max_batchsize=-1)
        self.model: DiffusionGenerator = model
        self.ema_model = torch.optim.swa_utils.AveragedModel(
            self.model,
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(ema_rate)
        )
        self.ema_model.requires_grad_(False)  # Ensure no gradient updates for EMA model
        self.gaussian_diffusion = gaussian_diffusion

        if sampler == "ddpm":
            self.sampler = DDPMSampler(self.gaussian_diffusion)
        else:
            self.sampler = DDIMSampler(self.gaussian_diffusion)

        self.lr = lr
        self.lr_warmup = lr_warmup

        self.n_samples = n_samples
        self.max_batchsize = max_batchsize

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

    def forward(self, gt_data: torch.Tensor, diffusion_steps: torch.Tensor, mask: torch.Tensor = None, emb: Dict = None,
                cond_data: torch.Tensor = None) -> Tuple[Dict[str, Tensor], Tensor]:
        """
        Forward pass through the model for training loss computation.

        :param gt_data: Ground truth data.
        :param diffusion_steps: Steps of the diffusion process.
        :param mask: Mask data. Defaults to None.
        :param cond_data: Conditioning data. Defaults to None.
        :param emb: embedding dictionary

        :return: Dictionary containing loss values for the training step and generated tensor.
        """
        kwargs = {"cond": cond_data}
        return self.gaussian_diffusion.training_losses(self.model, gt_data, diffusion_steps, mask, emb, **kwargs)

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        """
        Performs a single training step, calculates losses, and logs them.

        :param batch: Batch of input data.
        :param batch_idx: Index of the current batch.

        :return: Calculated loss for the current batch.
        """
        source, target, coords_input, coords_output, indices, mask, emb = batch
        diffusion_steps, weights = self.gaussian_diffusion.get_diffusion_steps(target.shape[0], target.device)
        l_dict, _ = self(target, diffusion_steps, mask, emb, source)
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
        source, target, coords_input, coords_output, indices, mask, emb = batch
        loss_dict = {}
        loss = []

        # Iterate over batch items and compute validation loss for each
        for i in range(target.shape[0]):
            diff_steps = self.gaussian_diffusion.diffusion_steps
            n_samples = 4
            t = torch.tensor([(diff_steps // n_samples) * x for x in range(n_samples - 1)] + [diff_steps-1]).to(target.device)
            source = torch.stack(n_samples * [source[i]])
            emb = {"CoordinateEmbedder": torch.stack(n_samples * [emb["CoordinateEmbedder"][i]]),
                   "VariableEmbedder": torch.stack(n_samples * [emb["VariableEmbedder"][i]])}
            l_dict, output = self(torch.stack(n_samples * [target[i]]), t, torch.stack(n_samples * [mask[i]]), emb, source)

            for k, v in l_dict.items():
                for ti in range(t.shape[0]):
                    loss_dict[f'val/step_{t[ti].item()}_{k}'] = v[ti]
                loss.append(v.mean())

            if batch_idx == 0 and i == 0:
                self.log_tensor_plot(torch.stack(n_samples * [target[i]]), torch.stack(n_samples * [source[i]]),
                                         output, coords_output, coords_input, f"tensor_plot_{self.current_epoch}")

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
            try:
                filename = os.path.join(save_dir, f"{plot_name}_{c}.png")
                self.logger.log_image(f"plots/{plot_name}", [filename])
            except Exception:
                pass

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

    def _predict_step(self, source, mask, emb, coords_input, coords_output, indices_sample):
        return self.sampler.sample_loop(self.model, source, mask,
                                        progress=True, emb=emb, coords_input=coords_input,
                                        coords_output=coords_output, indices_sample=indices_sample)
