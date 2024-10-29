import json
import os.path

import hydra
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig):
    # Save the composed configuration to a file
    if not os.path.exists(cfg.trainer.default_root_dir):
        os.makedirs(cfg.trainer.default_root_dir)
    composed_config_path = '{}/composed_config.yaml'.format(cfg.trainer.default_root_dir)
    with open(composed_config_path, 'w') as file:
        OmegaConf.save(config=cfg, f=file)

    # define data sets
    with open(cfg.dataloader.dataset.data_dict) as json_file:
        data = json.load(json_file)
    train_dataset = instantiate(cfg.dataloader.dataset, data_dict=data["train"])
    val_dataset = instantiate(cfg.dataloader.dataset, data_dict=data["val"])

    train_dataloader: DataLoader = instantiate(cfg.dataloader.dataloader, dataset=train_dataset)
    val_dataloader: DataLoader = instantiate(cfg.dataloader.dataloader, dataset=val_dataset)

    logger: WandbLogger = instantiate(cfg.logger)

    print(len(train_dataset))

    model = instantiate(cfg.model)
    trainer: Trainer = instantiate(cfg.trainer, logger=logger)

    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
                ckpt_path=cfg.ckpt_path)


if __name__ == "__main__":
    train()
