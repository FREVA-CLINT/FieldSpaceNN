import os
import hydra
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger, MLFlowLogger
from pytorch_lightning.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf
import torch
from ..src.utils.helpers import load_from_state_dict,freeze_zoom_levels,freeze_params
from ..src.data.pl_data_module import DataModule
import getpass
import mlflow

torch.manual_seed(42)


@hydra.main(version_base=None, config_path="/Users/maxwitte/work/stableclimgen/stableclimgen/configs/", config_name="era5_prediction_train")
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

    OmegaConf.set_struct(cfg, False)
    logger_conf = cfg.logger.logger
    if "WandbLogger" in logger_conf['_target_']:
        if not hasattr(logger_conf, "id") or logger_conf.id is None or (hasattr(cfg, "ckpt_path_pretrained") and cfg.ckpt_path_pretrained is not None):
            logger: WandbLogger = instantiate(logger_conf, id=None)
            if rank_zero_only.rank == 0:
                logger_conf.id = logger.experiment.id
        else:
            logger: WandbLogger = instantiate(logger_conf)
    elif "MLFlowLogger" in logger_conf['_target_']:
        if rank_zero_only.rank == 0:
            mlflow.enable_system_metrics_logging()
            mlflow.set_tracking_uri(logger_conf.tracking_uri)
            mlflow.set_experiment(cfg.project_name)
            mlflow.start_run(run_name=cfg.run_name, run_id=logger_conf.run_id, tags={"user": getpass.getuser()})
            if not hasattr(logger_conf, "run_id") or not logger_conf.run_id:
                logger_conf.run_id = mlflow.active_run().info.run_id
            mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
        logger: MLFlowLogger = instantiate(logger_conf)

    if rank_zero_only.rank == 0:
        # Create YAML config of training configuration
        composed_config_path = f'{cfg.trainer.default_root_dir}/composed_config.yaml'
        with open(composed_config_path, 'w') as file:
            OmegaConf.save(config=cfg, f=file)
    
    # Initialize model and trainer  
    model: any = instantiate(cfg.model)
    trainer: Trainer = instantiate(cfg.trainer, logger=logger)

    if rank_zero_only.rank == 0 and "WandbLogger" in logger_conf['_target_']:
        # Log model config
        logger.experiment.config.update(OmegaConf.to_container(
            cfg.model, resolve=True, throw_on_missing=False
        ), allow_val_change=True)

    model.log_images = cfg.logger.get('log_images', False)
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
