import json
import os
from typing import Any

import hydra
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf
from .utils.pl_data_module import DataModule
import torch

torch.manual_seed(42)


@hydra.main(version_base=None, config_path="../configs/", config_name="mgno_transformer_train")
def train(cfg: DictConfig) -> None:
    """
    Main training function that initializes datasets, dataloaders, model, and trainer,
    then begins the training process.

    :param cfg: Configuration object containing all settings for training, datasets,
                model, and logging.
    """
    # Ensure the default root directory exists, then save the configuration file
    if not os.path.exists(cfg.trainer.default_root_dir):
        os.makedirs(cfg.trainer.default_root_dir)

    # Create YAML config of training configuration
    composed_config_path = f'{cfg.trainer.default_root_dir}/composed_config.yaml'
    with open(composed_config_path, 'w') as file:
        OmegaConf.save(config=cfg, f=file)


    # Load data configuration and initialize datasets
    with open(cfg.dataloader.dataset.data_dict) as json_file:
        data = json.load(json_file)
    train_dataset = instantiate(cfg.dataloader.dataset, data_dict=data["train"])
    val_dataset = instantiate(cfg.dataloader.dataset, data_dict=data["val"])


    # Initialize the logger (e.g., Weights & Biases)
    logger: WandbLogger = instantiate(cfg.logger)
    
    # Initialize model and trainer  
    model: Any = instantiate(cfg.model)
    trainer: Trainer = instantiate(cfg.trainer, logger=logger)
    
    if rank_zero_only.rank == 0:
        # Log model config
        logger.experiment.config.update(OmegaConf.to_container(
            cfg.model, resolve=True, throw_on_missing=False
        ))

    data_module: DataModule = instantiate(cfg.dataloader.datamodule, train_dataset, val_dataset)

    # Start the training process
    trainer.fit(model=model, datamodule=data_module, ckpt_path=cfg.ckpt_path)


if __name__ == "__main__":
    train()
