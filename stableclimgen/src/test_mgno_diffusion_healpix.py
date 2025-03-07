import json
import os
from typing import Any

import hydra
import torch
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only

from stableclimgen.src.utils.pl_data_module import DataModule

torch.manual_seed(42)


def get_data(ds, ts, variables): 

    ds = ds.isel(time=ts)

    data_g = []
    for variable in variables:
        data = torch.tensor(ds[variable].values)
        data = data[0][0] if data.dim() > 2  else data
        data = data[0] if data.dim() > 1  else data
        data_g.append(data)

    data_g = torch.stack(data_g, dim=-1)

    ds.close()

    return data_g


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
    output, mask = trainer.predict(model=model, dataloaders=data_module.test_dataloader(), ckpt_path=cfg.ckpt_path)
    output = torch.cat(output)
    output = output.view(*output.shape[:3], -1)
    mask = torch.cat(mask)
    mask = output.view(*mask.shape[:3], -1)

    for k, var in enumerate(test_dataset.variables_target):
        output[:, :, :, k] = test_dataset.var_normalizers[var].denormalize(output[:, :, :, k])

    output = dict(zip(test_dataset.variables_target, output.split(1, dim=-2)))
    torch.save(output, cfg.output_path)
    mask = dict(zip(test_dataset.variables_target, mask.split(1, dim=-2)))
    torch.save(mask, cfg.output_path)

if __name__ == "__main__":
    test()
