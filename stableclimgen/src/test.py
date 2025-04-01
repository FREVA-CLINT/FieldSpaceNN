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
                               variables_source=data["train"]["source"]["variables"],
                               variables_target = data["train"]["target"]["variables"])

    # Initialize model and trainer
    model: Any = instantiate(cfg.model)
    trainer: Trainer = instantiate(cfg.trainer)

    data_module: DataModule = instantiate(cfg.dataloader.datamodule, dataset_test=test_dataset)

    # Start the training process
    predictions = trainer.predict(model=model, dataloaders=data_module.test_dataloader(), ckpt_path=cfg.ckpt_path)
    # Aggregate outputs from multiple devices
    output = torch.cat([batch["output"] for batch in predictions], dim=0)
    output = rearrange(output, "(b2 b1) n s ... -> b2 n (b1 s) ... ", b1=test_dataset.global_cells_input.shape[0])
    mask = torch.cat([batch["mask"] for batch in predictions], dim=0)
    mask = rearrange(mask, "(b2 b1) n s ... -> b2 n (b1 s) ... ", b1=test_dataset.global_cells_input.shape[0])

    for k, var in enumerate(test_dataset.variables_target):
        output[..., k] = test_dataset.var_normalizers[var].denormalize(output[..., k])

    output = dict(zip(test_dataset.variables_target, output.split(1, dim=-1)))
    torch.save(output, cfg.output_path)
    mask = dict(zip(test_dataset.variables_target, mask.split(1, dim=-1)))
    torch.save(mask, cfg.output_path.replace(".pt", "_mask.pt"))

if __name__ == "__main__":
    test()
