import getpass
import os
from typing import Any, Dict, List, Mapping, Optional

import mlflow
import torch
from lightning.pytorch.loggers import WandbLogger, MLFlowLogger, Logger
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import OmegaConf

from .visualization import healpix_plot_zooms_var, regular_plot
from ..modules.grids.grid_utils import decode_zooms


class CustomImageLogger(Logger):
    """
    A custom PyTorch Lightning logger that wraps either WandbLogger or MLFlowLogger
    and adds a method for logging image plots.

    It's instantiated with a `logger_type` ('wandb' or 'mlflow') and passes
    other keyword arguments to the underlying logger.
    """

    def __init__(
        self,
        cfg: Optional[Dict[str, Any]] = None,
        logger_type: str = 'wandb',
        save_snapshot_images: bool = True,
        log_snapshot_images: bool = True,
        **kwargs: Any
    ):
        """
        Initialize the custom image logger wrapper.

        :param cfg: Configuration dictionary.
        :param logger_type: Backend logger type ("wandb" or "mlflow").
        :param save_snapshot_images: Whether to save images locally.
        :param log_snapshot_images: Whether to log images to the backend.
        :param kwargs: Additional keyword arguments forwarded to the backend logger.
        :return: None.
        """
        super().__init__()
        self.save_snapshot_images: bool = save_snapshot_images
        self.log_snapshot_images: bool = log_snapshot_images
        self.cfg: Optional[Dict[str, Any]] = cfg
        self.logger_conf: Dict[str, Any] = kwargs
        self._internal_logger: Logger

        OmegaConf.set_struct(cfg, False)
        if logger_type == 'wandb':
            # Handle WandB run resuming logic
            ckpt_path = self.cfg.get("ckpt_path_pretrained")
            run_id = self.logger_conf.get("id")
            if not run_id or (ckpt_path is not None):
                self._internal_logger = WandbLogger(**self.logger_conf, id=None)
                if rank_zero_only.rank == 0:
                    # Save the new run id to the config for other processes
                    OmegaConf.update(self.cfg, f"logger.id", self._internal_logger.experiment.id, merge=True)
            else:
                self._internal_logger = WandbLogger(**self.logger_conf)

        elif logger_type == 'mlflow':
            # Handle MLFlow setup and run starting
            if rank_zero_only.rank == 0:
                mlflow.enable_system_metrics_logging()
                mlflow.set_tracking_uri(self.logger_conf.get("tracking_uri"))
                mlflow.set_experiment(self.cfg.get("project_name"))
                run_id = self.logger_conf.get("run_id")
                mlflow.start_run(run_name=self.cfg.get("run_name"), run_id=run_id, tags={"user": getpass.getuser()})
                if not run_id:
                    new_run_id = mlflow.active_run().info.run_id
                    # Save the new run id to the config for other processes
                    OmegaConf.update(self.cfg, f"logger.run_id", new_run_id, merge=True)
                
                # Log all parameters
                mlflow.log_params(OmegaConf.to_container(self.cfg, resolve=True))
            
            self._internal_logger = MLFlowLogger(**self.logger_conf)
        else:
            raise ValueError(f"Unsupported logger_type: '{logger_type}'. Choose 'wandb' or 'mlflow'.")
        
        # Create and save the full training configuration file
        if rank_zero_only.rank == 0:
            # Create YAML config of training configuration
            composed_config_path = f'{cfg.trainer.default_root_dir}/composed_config.yaml'
            with open(composed_config_path, 'w') as file:
                OmegaConf.save(config=cfg, f=file)


    @property
    def experiment(self):
        return self._internal_logger.experiment

    @property
    def name(self):
        return self._internal_logger.name

    @property
    def version(self):
        return self._internal_logger.version

    @rank_zero_only
    def log_hyperparams(self, params: Mapping[str, Any], *args: Any, **kwargs: Any):
        """
        Log hyperparameters to the backend logger.

        :param params: Hyperparameter mapping.
        :param args: Additional positional arguments.
        :param kwargs: Additional keyword arguments.
        :return: None.
        """
        self._internal_logger.log_hyperparams(params, *args, **kwargs)

        # Also log model config to wandb if it's the backend
        if isinstance(self._internal_logger, WandbLogger):
            self.experiment.config.update(OmegaConf.to_container(
                self.cfg.get('model', {}), resolve=True, throw_on_missing=False
            ), allow_val_change=True)


    @rank_zero_only
    def log_metrics(self, metrics: Mapping[str, float], step: int):
        """
        Log scalar metrics to the backend logger.

        :param metrics: Metric dictionary.
        :param step: Global step index.
        :return: None.
        """
        self._internal_logger.log_metrics(metrics, step)

    def log_healpix_tensor_plot(
        self,
        input_data: Dict[int, torch.Tensor],
        output: Optional[Dict[int, torch.Tensor]],
        gt: Dict[int, torch.Tensor],
        mask: Optional[Dict[int, torch.Tensor]],
        sample_configs: Dict[int, Dict[str, Any]],
        emb: Optional[Dict[str, Any]],
        max_zoom: int,
        current_epoch: int,
        output_comp: Optional[Dict[int, torch.Tensor]] = None,
        plot_name: str = ""
    ):
        """
        Generates and logs plots of input, output, and ground truth tensors.

        :param input_data: Input tensor dict by zoom with shape ``(b, v, t, n, d, f)``.
        :param output: Output tensor dict by zoom with shape ``(b, v, t, n, d, f)``.
        :param gt: Ground-truth tensor dict by zoom with shape ``(b, v, t, n, d, f)``.
        :param mask: Optional mask dict by zoom with shape ``(b, v, t, n, d, m)``.
        :param sample_configs: Sampling configuration per zoom.
        :param emb: Optional embedding dictionary.
        :param max_zoom: Maximum zoom level for composite plots.
        :param current_epoch: Current epoch index for naming.
        :param output_comp: Optional composite output dict by zoom.
        :param plot_name: Optional suffix for plot names.
        :return: None.
        """
        if not self.save_snapshot_images:
            return

        if isinstance(self._internal_logger, WandbLogger):
            save_dir = os.path.join(self._internal_logger.save_dir, "validation_images")
        elif isinstance(self._internal_logger, MLFlowLogger):
            if mlflow.active_run():
                save_dir = os.path.join(mlflow.get_artifact_uri().replace("file://", ""), "validation_images")
            else:
                save_dir = "validation_images"
        else:
            save_dir = "validation_images"

        os.makedirs(save_dir, exist_ok=True)

        save_paths = []

        if output is not None:
            save_paths += healpix_plot_zooms_var(
                input_data, output, gt, save_dir, mask_zooms=mask, sample_configs=sample_configs,
                plot_name=f"epoch_{current_epoch}{plot_name}", emb=emb,
            )

        # Build combined plots at the maximum zoom for easier visual comparison.
        source_p = decode_zooms(input_data, sample_configs=sample_configs, out_zoom=max_zoom)
        target_p = decode_zooms(gt, sample_configs=sample_configs, out_zoom=max_zoom)

        output_p = output_comp

        mask_p = {max_zoom: mask[max_zoom]} if mask is not None and max_zoom in mask else None
        save_paths += healpix_plot_zooms_var(
            source_p, output_p, target_p, save_dir, mask_zooms=mask_p, sample_configs=sample_configs,
            plot_name=f"epoch_{current_epoch}_combined{plot_name}", emb=emb,
        )

        if self.log_snapshot_images:
            for save_path in save_paths:
                if isinstance(self._internal_logger, WandbLogger):
                    self._internal_logger.log_image(f"plots/{os.path.basename(save_path).replace('.png', '')}", [save_path])
                elif isinstance(self._internal_logger, MLFlowLogger) and mlflow.active_run():
                    mlflow.log_artifact(save_path, artifact_path="plots")

    def log_regular_tensor_plot(
        self,
        gt_tensor: torch.Tensor,
        in_tensor: torch.Tensor,
        rec_tensor: torch.Tensor,
        target_coords: torch.Tensor,
        in_coords: torch.Tensor,
        plot_name: str,
    ):
        """
        Logs a plot of ground truth and reconstructed tensor images for visualization.

        :param gt_tensor: Ground truth tensor of shape ``(b, v, t, n, d, f)`` or a
            CNN-friendly view such as ``(b, c, h, w)``.
        :param in_tensor: Input tensor aligned with ``gt_tensor``.
        :param rec_tensor: Reconstructed tensor aligned with ``gt_tensor``.
        :param target_coords: Ground truth coordinates tensor.
        :param in_coords: Input coordinates tensor.
        :param plot_name: Name for the plot to be saved.
        :return: None.
        """
        if not self.save_snapshot_images:
            return
        
        if isinstance(self._internal_logger, WandbLogger):
            save_dir = os.path.join(self._internal_logger.save_dir, "validation_images")
        elif isinstance(self._internal_logger, MLFlowLogger):
            if mlflow.active_run():
                save_dir = os.path.join(mlflow.get_artifact_uri().replace("file://", ""), "validation_images")
            else:
                save_dir = "validation_images"
        else:
            save_dir = "validation_images"

        os.makedirs(save_dir, exist_ok=True)

        save_paths = regular_plot(gt_tensor, in_tensor, rec_tensor, plot_name, save_dir, target_coords, in_coords)

        if self.log_snapshot_images:
            for save_path in save_paths:
                if isinstance(self._internal_logger, WandbLogger):
                    self._internal_logger.log_image(f"plots/{os.path.basename(save_path).replace('.png', '')}", [save_path])
                elif isinstance(self._internal_logger, MLFlowLogger) and mlflow.active_run():
                    mlflow.log_artifact(save_path, artifact_path="plots")
