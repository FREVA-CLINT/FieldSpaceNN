import os

import lightning.pytorch as pl
import torch
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .model import DiffusionGenerator
from ...utils.visualization import plot_images


class LightningDiffusionGenerator(pl.LightningModule):
    def __init__(self, model, gaussian_diffusion, lr, lr_warmup=None, ema_rate=0.999):
        super().__init__()
        self.model: DiffusionGenerator = model
        self.ema_model = torch.optim.swa_utils.AveragedModel(
            self.model,
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(
                ema_rate
            )
        )
        self.ema_model.requires_grad_(False)
        self.gaussian_diffusion = gaussian_diffusion
        self.lr = lr
        self.lr_warmup = lr_warmup
        self.save_hyperparameters(ignore=['model'])

    def on_before_zero_grad(self, optimizer):
        self.ema_model.update_parameters(self.model)

    def on_train_end(self):
        torch.optim.swa_utils.update_bn(
            self.trainer.train_dataloader, self.ema_model
        )

    def forward(self, gt_data, diffusion_steps, mask_data=None, cond_data=None, coords=None):
        return self.gaussian_diffusion.training_losses(
            self.model, gt_data, diffusion_steps, mask_data, cond_data, coords
        )

    def training_step(self, batch, batch_idx):
        cond_data, cond_coords, gt_data, gt_coords, mask_data = batch
        diffusion_steps, weights = self.gaussian_diffusion.get_diffusion_steps(gt_data.shape[0], gt_data.device)
        l_dict, _ = self(gt_data, diffusion_steps, mask_data, cond_data, gt_coords)
        loss = (l_dict["loss"] * weights).mean()
        loss_dict = {}
        for k, v in l_dict.items():
            loss_dict['train/{}'.format(k)] = v.mean()
        self.log_dict(loss_dict, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        cond_data, cond_coords, gt_data, gt_coords, mask_data = batch
        loss_dict = {}
        loss = []
        for i in range(gt_data.shape[0]):
            diff_steps = self.gaussian_diffusion.diffusion_steps
            t = torch.tensor([(diff_steps // 10) * x for x in range(10)]).to(gt_data.device)
            cond = torch.stack(10 * [cond_data[i]])
            l_dict, output = self(torch.stack(10 * [gt_data[i]]), t, torch.stack(10 * [mask_data[i]]), cond, gt_coords)
            for k, v in l_dict.items():
                for ti in range(t.shape[0]):
                    loss_dict['val/step_{}_{}'.format(t[ti].item(), k)] = v[ti]
                loss.append(v.mean())
            if batch_idx == 0 and i == 0:
                try:
                    self.log_tensor_plot(torch.stack(10 * [gt_data[i]]), output,
                                         f"tensor_plot_{self.current_epoch}")
                except Exception as e:
                    pass
        self.log_dict(loss_dict, sync_dist=True)
        self.log('val_loss', torch.stack(loss).mean(), sync_dist=True)
        return self.log_dict

    def log_tensor_plot(self, gt_tensor, rec_tensor, plot_name):
        # Define the directory to save the metrics
        save_dir = os.path.join(self.trainer.logger.save_dir, "validation_images")
        os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
        plot_images(gt_tensor, rec_tensor, f"{plot_name}", save_dir)
        for c in range(gt_tensor.shape[1]):
            filename = os.path.join(save_dir, f"{plot_name}_{c}.png")

            # Log the plot to W&B
            self.logger.log_image(f"plots/{plot_name}", [filename])

    def configure_optimizers(self):
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
            def constant_lr_lambda(epoch):
                return 1.0
            scheduler = LambdaLR(optimizer, lr_lambda=constant_lr_lambda)

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
