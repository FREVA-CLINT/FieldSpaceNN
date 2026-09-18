import os
from collections.abc import Mapping

import hydra
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf
import torch
from ..src.utils.helpers import load_pretrained_checkpoints, freeze_zoom_levels
from ..src.data.pl_data_module import DataModule
from ..src.data.datasets_healpix_in_memory import _resolve_distributed_context

import logging
logging.getLogger("fsspec.reference").setLevel(logging.WARNING)

torch.manual_seed(42)


def _rank_sharding_requested(cfg: DictConfig) -> bool:
    """Return whether all configured train/validation datasets request rank sharding."""
    train_config = cfg.dataloader.dataset
    train_sharded = bool(train_config.get("distributed_shard", False))

    val_config = cfg.dataloader.get("val_dataset")
    if val_config is None:
        return train_sharded

    val_sharded = bool(val_config.get("distributed_shard", False))
    if train_sharded != val_sharded:
        raise ValueError(
            "Training and validation datasets must either both enable or both disable "
            "`distributed_shard`."
        )
    return train_sharded


def _is_standard_ddp_strategy(strategy: object) -> bool:
    """Return whether a Hydra Trainer strategy selects standard popen DDP."""
    if isinstance(strategy, str):
        return strategy == "ddp"
    if isinstance(strategy, Mapping):
        target = str(strategy.get("_target_", ""))
        start_method = strategy.get("start_method", "popen")
        return target.endswith(".DDPStrategy") and start_method == "popen"
    return False


def _validate_rank_sharded_training_config(cfg: DictConfig) -> None:
    """Fail before preload when rank-local datasets and Trainer settings conflict."""
    if not _rank_sharding_requested(cfg):
        return

    if not _is_standard_ddp_strategy(cfg.trainer.get("strategy")):
        raise ValueError(
            "`distributed_shard=true` requires ordinary `trainer.strategy=ddp`; "
            "fork, spawn, and automatic self-launch strategies are unsupported."
        )
    if bool(cfg.trainer.get("use_distributed_sampler", True)):
        raise ValueError(
            "`distributed_shard=true` requires "
            "`trainer.use_distributed_sampler=false` because each process already "
            "owns its local dataset items."
        )
    if bool(cfg.dataloader.datamodule.get("use_costum_ddp_sampler", False)):
        raise ValueError(
            "`distributed_shard=true` requires "
            "`dataloader.datamodule.use_costum_ddp_sampler=false`."
        )

    plugins = cfg.trainer.get("plugins")
    if plugins is not None:
        plugins_text = str(OmegaConf.to_container(plugins, resolve=True))
        if "LightningEnvironment" in plugins_text:
            raise ValueError(
                "Rank-sharded Slurm DDP must use Lightning's native SLURMEnvironment. "
                "Remove the configured LightningEnvironment plugin."
            )

    _, world_size = _resolve_distributed_context()
    devices = cfg.trainer.get("devices")
    num_nodes = int(cfg.trainer.get("num_nodes", 1))
    if not isinstance(devices, int) or devices <= 0:
        raise ValueError(
            "Rank-sharded DDP requires an integer `trainer.devices` matching the "
            "number of Slurm tasks per node."
        )
    expected_world_size = devices * num_nodes
    if expected_world_size != world_size:
        raise ValueError(
            f"Trainer expects {devices} devices x {num_nodes} nodes = "
            f"{expected_world_size} ranks, but the external environment reports "
            f"WORLD_SIZE/SLURM_NTASKS={world_size}."
        )


def _instantiate_validation_dataset(cfg: DictConfig):
    val_config = cfg.dataloader.get("val_dataset")
    if val_config is not None:
        return instantiate(val_config, data_dict=cfg.data_split['val'])
    return instantiate(cfg.dataloader.dataset, data_dict=cfg.data_split['val'])


@hydra.main(version_base=None, config_path="../configs/", config_name="era5_prediction_flowmatching")
def train(cfg: DictConfig) -> None:
    """
    Main training function that initializes datasets, dataloaders, model, and trainer,
    then begins the training process. 

    :param cfg: Configuration object containing all settings for training, datasets,
                model, and logging.
    """
    rank_sharded = _rank_sharding_requested(cfg)
    _validate_rank_sharded_training_config(cfg)

    # Ensure the default root directory exists, then save the configuration file
    if rank_zero_only.rank == 0 and not os.path.exists(cfg.trainer.default_root_dir):
        os.makedirs(cfg.trainer.default_root_dir)

    if rank_sharded:
        # Validation is normally much smaller. Registering it first prevents the
        # train cache from being held beside a second full train/validation union.
        val_dataset = _instantiate_validation_dataset(cfg)
        train_dataset = instantiate(cfg.dataloader.dataset, data_dict=cfg.data_split['train'])
    else:
        train_dataset = instantiate(cfg.dataloader.dataset, data_dict=cfg.data_split['train'])
        val_dataset = _instantiate_validation_dataset(cfg)
    
    # initialize logger, model and trainer
    logger = instantiate(cfg.logger, cfg=cfg, _recursive_=False)
    model: any = instantiate(cfg.model)
    trainer: Trainer = instantiate(cfg.trainer, logger=logger)

    data_module: DataModule = instantiate(cfg.dataloader.datamodule, train_dataset, val_dataset)
    
    if hasattr(cfg, "ckpt_path_pretrained") and cfg.ckpt_path_pretrained is not None:
        model, _ = load_pretrained_checkpoints(
            model,
            cfg.ckpt_path_pretrained,
            freeze_pretrained=getattr(cfg, "freeze_pretrained", False),
            device="cpu",
            print_keys=True,
        )

    if 'freeze_zooms' in cfg.keys():
        freeze_zoom_levels(model, cfg.freeze_zooms)

    # Start the training process
    trainer.fit(model=model, datamodule=data_module, ckpt_path=cfg.ckpt_path)


if __name__ == "__main__":
    train()
