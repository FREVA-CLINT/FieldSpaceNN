import os
import hydra
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from pytorch_lightning.utilities import rank_zero_only
from omegaconf import DictConfig
import torch
from ..src.utils.helpers import load_from_state_dict,freeze_zoom_levels,freeze_params
from ..src.data.pl_data_module import DataModule

torch.manual_seed(42)


@hydra.main(version_base=None, config_path="../configs/", config_name="era5_prediction_train")
def train(cfg: DictConfig) -> None:
    """
    Main training function that initializes datasets, dataloaders, model, and trainer,
    then begins the training process. 

    :param cfg: Configuration object containing all settings for training, datasets,
                model, and logging.
    """
    # Ensure the default root directory exists, then save the configuration file
    if rank_zero_only.rank == 0 and not os.path.exists(cfg.trainer.default_root_dir):
        os.makedirs(cfg.trainer.default_root_dir)

    train_dataset = instantiate(cfg.dataloader.dataset, data_dict=cfg.data_split['train'])
    if hasattr(cfg.dataloader,'val_dataset') and cfg.dataloader.val_dataset is not None:
        val_dataset = instantiate(cfg.dataloader.val_dataset, data_dict=cfg.data_split['val'])
    else:
        val_dataset = instantiate(cfg.dataloader.dataset, data_dict=cfg.data_split['val'])
    
    # initialize logger, model and trainer
    logger = instantiate(cfg.logger, cfg=cfg, _recursive_=False)
    model: any = instantiate(cfg.model)
    trainer: Trainer = instantiate(cfg.trainer, logger=logger)

    data_module: DataModule = instantiate(cfg.dataloader.datamodule, train_dataset, val_dataset)
    
    if hasattr(cfg, "ckpt_path_pretrained") and cfg.ckpt_path_pretrained is not None:
        model, matching_keys = load_from_state_dict(model, cfg.ckpt_path_pretrained, print_keys=True)
        if hasattr(cfg, "freeze_pretrained") and cfg.freeze_pretrained:
            freeze_params(model, matching_keys)

    if 'freeze_zooms' in cfg.keys():
        freeze_zoom_levels(model, cfg.freeze_zooms)

    # Start the training process
    trainer.fit(model=model, datamodule=data_module, ckpt_path=cfg.ckpt_path)


if __name__ == "__main__":
    train()
