import json
import os
from typing import Any

import hydra
import torch
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from omegaconf import DictConfig
from einops import rearrange
import xarray as xr
import numpy as np

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
    output = rearrange(output, "b n t s ... -> (b t) n s ... ")
    if test_dataset.norm_dict and output.dim() == 5:
        for k, var in enumerate(test_dataset.variables_target):
            output[..., k] = test_dataset.var_normalizers[var].denormalize(output[..., k])

    output_dict = dict(zip(test_dataset.variables_target, output.split(1, dim=3)))
    if test_dataset.sample_timesteps:
        time = xr.open_dataset(test_dataset.files_source[0], decode_times=False)["time"][test_dataset.sample_timesteps]
    else:
        time = xr.open_dataset(test_dataset.files_source[0], decode_times=False)["time"][:]
    if "export_to_zarr" in cfg.keys() and cfg.export_to_zarr:
        for sample in range(output.shape[1]):
            data_vars = {}
            for var, data in output_dict.items():
                xr_data = data[:, sample, :, 0].clone().detach().cpu().numpy()
                data_vars[var] = (["time", "cell", "level"], xr_data)

            coords = {
                "time": time.values,
                "level": range(data.shape[-1]),  # or use your actual sample indices
                "cell": np.arange(data.shape[2]),  # Should be a 1D array of cell indices or IDs
            }

            ds = xr.Dataset(data_vars=data_vars, coords=coords)
            ds.to_zarr(cfg.output_path, mode="w")

if __name__ == "__main__":
    test()
