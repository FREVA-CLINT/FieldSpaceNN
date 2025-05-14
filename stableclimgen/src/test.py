import json
import os
from typing import Any

import hydra
import torch
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from omegaconf import DictConfig
from einops import rearrange

from stableclimgen.src.utils.pl_data_module import DataModule


@hydra.main(version_base=None, config_path="../configs/", config_name="healpix_vae_test")
def test(cfg: DictConfig) -> None:
    """
    Main training function that initializes datasets, dataloaders, model, and trainer,
    then begins the training process.

    :param cfg: Configuration object containing all settings for training, datasets,
                model, and logging.
    """
    if not os.path.exists(os.path.dirname(cfg.output_path)):
        os.makedirs(os.path.dirname(cfg.output_path))

    # Load data configuration and initialize datasets
    with open(cfg.dataloader.dataset.data_dict) as json_file:
        data = json.load(json_file)
    test_dataset = instantiate(cfg.dataloader.dataset,
                               data_dict=data["test"],
                               variables_source = data["train"]["source"]["variables"],
                               variables_target = data["train"]["target"]["variables"])

    # Initialize model and trainer
    model: Any = instantiate(cfg.model)
    trainer: Trainer = instantiate(cfg.trainer)

    data_module: DataModule = instantiate(cfg.dataloader.datamodule, dataset_test=test_dataset)

    # Start the training process
    predictions = trainer.predict(model=model, dataloaders=data_module.test_dataloader(), ckpt_path=cfg.ckpt_path)
    # Aggregate outputs from multiple devices
    output = torch.cat([batch["output"] for batch in predictions], dim=0)
    output = rearrange(output, "(b2 b1) n t s ... -> b2 n t (b1 s) ... ",
                       b1=test_dataset.global_cells_input.shape[0] if hasattr(test_dataset,
                                                                              "global_cells_input") else 1)
    mask = torch.cat([batch["mask"] for batch in predictions], dim=0)
    mask = rearrange(mask, "(b2 b1) n t s ... -> b2 n t (b1 s) ... ",
                     b1=test_dataset.global_cells_input.shape[0] if hasattr(test_dataset, "global_cells_input") else 1)

    if test_dataset.norm_dict:
        for k, var in enumerate(test_dataset.variables_target):
            output[..., k] = test_dataset.var_normalizers[var].denormalize(output[..., k])

    output = dict(zip(test_dataset.variables_target, output.split(1, dim=-1)))
    torch.save(output, cfg.output_path)
    mask = dict(zip(test_dataset.variables_target, mask.split(1, dim=-1)))
    torch.save(mask, cfg.output_path.replace(".pt", "_mask.pt"))

    if 'output_var' in predictions[0].keys() and predictions[0]['output_var'] is not None:
        output_var = torch.cat([batch["output_var"] for batch in predictions], dim=0)
        output_var = rearrange(output_var, "(b2 b1) n t s ... -> b2 n t (b1 s) ... ",
                               b1=test_dataset.global_cells_input.shape[0] if hasattr(test_dataset,
                                                                                      "global_cells_input") else 1)

        for k, var in enumerate(test_dataset.variables_target):
            output_var[..., k] = test_dataset.var_normalizers[var].denormalize_var(output_var[..., k])

        output_var = dict(zip(test_dataset.variables_target, output_var.split(1, dim=-1)))
        torch.save(output_var, cfg.output_path.replace(".pt", "_var.pt"))

if __name__ == "__main__":
    test()
