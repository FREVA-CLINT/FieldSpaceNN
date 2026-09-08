import getpass
import os
from typing import Any, Dict, List, Mapping, Optional

import mlflow
import torch
from lightning.pytorch.loggers import Logger, MLFlowLogger, WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf

from .visualization import regular_plot, healpix_plot_zooms_var, healpix_plot_zooms_time
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
        plot_types: Optional[List[str]] = None,
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
        self.plot_types: List[str] = plot_types or []
        self.logger_conf: Dict[str, Any] = kwargs
        self._internal_logger: Logger
        self._composed_config_path = os.path.join(
            str(cfg.trainer.default_root_dir), "composed_config.yaml"
        )
        self._mlflow_run_initialized = False
        self._owns_active_mlflow_run = False
        self._mlflow_log_system_metrics = False
        self._mlflow_system_metrics_sampling_interval: Optional[int] = None
        self._mlflow_system_metrics_samples_before_logging: Optional[int] = None

        OmegaConf.set_struct(cfg, False)
        if logger_type == 'wandb':
            # Handle WandB run resuming logic
            ckpt_path = self.cfg.get("ckpt_path_pretrained")
            run_id = self.logger_conf.get("id")
            if not run_id or (ckpt_path is not None):
                # Generate the ID without accessing ``WandbLogger.experiment``.
                # Accessing ``experiment`` starts the W&B background service, which
                # is unsafe before Lightning launches ``ddp_fork`` workers.
                from wandb.util import generate_id

                run_id = generate_id()
                fresh_logger_conf = dict(self.logger_conf)
                fresh_logger_conf.pop("id", None)
                self._internal_logger = WandbLogger(**fresh_logger_conf, id=run_id)
                if rank_zero_only.rank == 0:
                    # Save the new run id to the config for other processes
                    OmegaConf.update(self.cfg, "logger.id", run_id, merge=True)
            else:
                self._internal_logger = WandbLogger(**self.logger_conf)

        elif logger_type == 'mlflow':
            mlflow_conf = dict(self.logger_conf)
            workspace = mlflow_conf.pop("workspace", None)
            self._mlflow_log_system_metrics = bool(
                mlflow_conf.pop("log_system_metrics", False)
            )
            self._mlflow_system_metrics_sampling_interval = mlflow_conf.pop(
                "system_metrics_sampling_interval", None
            )
            self._mlflow_system_metrics_samples_before_logging = mlflow_conf.pop(
                "system_metrics_samples_before_logging", None
            )

            mlflow_conf.setdefault(
                "experiment_name", self.cfg.get("project_name", "lightning_logs")
            )
            mlflow_conf.setdefault("run_name", self.cfg.get("run_name"))

            tags = mlflow_conf.get("tags")
            if isinstance(tags, DictConfig):
                tags = OmegaConf.to_container(tags, resolve=True)
            tags = dict(tags or {})
            tags.setdefault("user", getpass.getuser())
            mlflow_conf["tags"] = tags

            tracking_uri = mlflow_conf.get("tracking_uri")
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            if workspace:
                if not hasattr(mlflow, "set_workspace"):
                    raise RuntimeError(
                        "MLflow workspace selection requires mlflow>=3.15.1; "
                        f"found {mlflow.__version__}."
                    )
                mlflow.set_workspace(workspace)

            # MLFlowLogger owns experiment creation, run resumption, metrics, and
            # checkpoint logging. The fluent run is attached lazily from save() or
            # the first logging call so system-metrics threads are not started
            # before Lightning launches distributed worker processes.
            self._internal_logger = MLFlowLogger(**mlflow_conf)
        else:
            raise ValueError(f"Unsupported logger_type: '{logger_type}'. Choose 'wandb' or 'mlflow'.")

        # Create and save the full training configuration file
        if rank_zero_only.rank == 0:
            self._save_composed_config()

    def _save_composed_config(self) -> None:
        """Persist the resolved Hydra configuration beside the checkpoints."""
        os.makedirs(os.path.dirname(self._composed_config_path), exist_ok=True)
        OmegaConf.save(config=self.cfg, f=self._composed_config_path, resolve=True)

    def _ensure_mlflow_run(self) -> None:
        """Initialize one MLflow run and attach the optional system monitor."""
        if not isinstance(self._internal_logger, MLFlowLogger):
            return
        if self._mlflow_run_initialized:
            return

        run_id = self._internal_logger.run_id
        if run_id is None:
            raise RuntimeError("MLflow did not create or resume a run.")

        self._mlflow_run_initialized = True
        self.logger_conf["run_id"] = run_id
        OmegaConf.update(self.cfg, "logger.run_id", run_id, merge=True)
        self._save_composed_config()

        if self._mlflow_log_system_metrics:
            if self._mlflow_system_metrics_sampling_interval is not None:
                mlflow.set_system_metrics_sampling_interval(
                    self._mlflow_system_metrics_sampling_interval
                )
            if self._mlflow_system_metrics_samples_before_logging is not None:
                mlflow.set_system_metrics_samples_before_logging(
                    self._mlflow_system_metrics_samples_before_logging
                )

            active_run = mlflow.active_run()
            if active_run is None:
                mlflow.start_run(run_id=run_id, log_system_metrics=True)
                self._owns_active_mlflow_run = True
            elif active_run.info.run_id != run_id:
                raise RuntimeError(
                    "Cannot attach MLflow system metrics: another run is already active "
                    f"({active_run.info.run_id})."
                )

        self._internal_logger.experiment.log_artifact(
            run_id,
            self._composed_config_path,
            artifact_path="config",
        )

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

        MLflow receives the instantiated architecture configuration from
        ``cfg.model.model`` because the Lightning modules intentionally exclude the
        wrapped model from ``save_hyperparameters``. Other backends receive the
        hyperparameters supplied by Lightning.

        :param params: Hyperparameter mapping supplied by Lightning.
        :param args: Additional positional arguments.
        :param kwargs: Additional keyword arguments.
        :return: None.
        """
        self._ensure_mlflow_run()
        backend_params = params
        if isinstance(self._internal_logger, MLFlowLogger):
            backend_params = OmegaConf.to_container(
                self.cfg.model.model,
                resolve=True,
                throw_on_missing=False,
            )
        self._internal_logger.log_hyperparams(backend_params, *args, **kwargs)

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
        self._ensure_mlflow_run()
        self._internal_logger.log_metrics(metrics, step)

    @rank_zero_only
    def save(self) -> None:
        """Flush the wrapped logger and ensure MLflow has a run before fitting."""
        self._ensure_mlflow_run()
        self._internal_logger.save()

    @rank_zero_only
    def finalize(self, status: str = "success") -> None:
        """Finalize the wrapped logger and stop MLflow system monitoring."""
        self._internal_logger.finalize(status)

        if self._owns_active_mlflow_run:
            active_run = mlflow.active_run()
            if active_run is not None and active_run.info.run_id == self.version:
                mlflow_status = {
                    "success": "FINISHED",
                    "finished": "FINISHED",
                    "failed": "FAILED",
                }.get(status, "KILLED")
                mlflow.end_run(status=mlflow_status)
            self._owns_active_mlflow_run = False

    @rank_zero_only
    def after_save_checkpoint(self, checkpoint_callback: Any) -> None:
        """Delegate checkpoint artifact handling to the selected backend."""
        self._internal_logger.after_save_checkpoint(checkpoint_callback)

    def _get_validation_image_dir(self) -> str:
        if isinstance(self._internal_logger, WandbLogger):
            save_dir = os.path.join(self._internal_logger.save_dir, "validation_images")
        elif isinstance(self._internal_logger, MLFlowLogger):
            local_root = self.logger_conf.get("save_dir") or self.cfg.trainer.default_root_dir
            save_dir = os.path.join(str(local_root), "validation_images")
        else:
            save_dir = "validation_images"

        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def _log_saved_paths(self, save_paths: List[str]):
        if not self.log_snapshot_images:
            return

        for save_path in save_paths:
            if isinstance(self._internal_logger, WandbLogger):
                self._internal_logger.log_image(
                    f"plots/{os.path.basename(save_path).replace('.png', '')}",
                    [save_path],
                )
            elif isinstance(self._internal_logger, MLFlowLogger):
                self._ensure_mlflow_run()
                self._internal_logger.experiment.log_artifact(
                    self.version,
                    save_path,
                    artifact_path="plots",
                )

    def log_tensor_plot(
        self,
        plot_types: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Dispatch tensor plotting to one or more configured plot functions.

        Supported plot types:
        - ``regular_plot``
        - ``healpix_plot_zooms_var``
        """
        if not self.save_snapshot_images:
            return []

        requested_plot_types = plot_types if plot_types is not None else self.plot_types
        if not requested_plot_types:
            return []

        save_dir = kwargs.get("save_dir", self._get_validation_image_dir())
        save_paths: List[str] = []

        for plot_type in requested_plot_types:
            if "regular" in plot_type:
                save_paths.extend(
                    regular_plot(
                        kwargs["gt"],
                        kwargs["input"],
                        kwargs["output"],
                        kwargs["plot_name"],
                        save_dir,
                        kwargs.get("target_coords"),
                        kwargs.get("in_coords"),
                    )
                )

            if "zooms" in plot_type:
                save_paths.extend(
                    healpix_plot_zooms_var(
                        kwargs["input"],
                        kwargs["output"],
                        kwargs["gt"],
                        save_dir,
                        mask_zooms=kwargs.get("mask"),
                        sample_configs=kwargs.get("sample_configs", {}),
                        emb=kwargs.get("emb"),
                        plot_name=kwargs.get("plot_name", "healpix_plot"),
                        sample=kwargs.get("sample", 0),
                        plot_n_vars=kwargs.get("plot_n_vars", -1),
                        plot_n_ts=kwargs.get("plot_n_ts", 1),
                    )
                )
            
            if "time" in plot_type:
                save_paths.extend(
                    healpix_plot_zooms_time(
                        kwargs["output"],
                        kwargs["gt"],
                        save_dir,
                        mask_zooms=kwargs.get("mask"),
                        sample_configs=kwargs.get("sample_configs", {}),
                        emb=kwargs.get("emb"),
                        plot_name=kwargs.get("plot_name", "healpix_plot"),
                        sample=kwargs.get("sample", 0),
                        plot_n_vars=kwargs.get("plot_n_vars", -1),
                        plot_n_ts=kwargs.get("plot_n_ts", 1),
                    )
                )

        self._log_saved_paths(save_paths)
        return save_paths

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
        plot_name: str = "",
        plot_combined: bool = True,
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
        :param plot_combined: Whether to decode zoom residuals and log a combined plot.
        :return: None.
        """
        if not self.save_snapshot_images:
            return

        if output is not None:
            self.log_tensor_plot(
                plot_types=["healpix_plot_zooms_var"],
                input=input_data,
                output=output,
                gt=gt,
                mask=mask,
                sample_configs=sample_configs,
                emb=emb,
                plot_name=f"epoch_{current_epoch}{plot_name}",
            )

        if not plot_combined:
            return

        # Build one combined plot by decoding each tensor dict to a shared zoom.
        combined_zoom_candidates = []
        for zoom_dict in (input_data, output, gt):
            if zoom_dict:
                combined_zoom_candidates.extend([int(z) for z in zoom_dict.keys()])
        combined_zoom = max(combined_zoom_candidates) if combined_zoom_candidates else max_zoom

        source_p = decode_zooms(input_data.copy(), sample_configs=sample_configs, out_zoom=combined_zoom) if input_data else {}
        target_p = decode_zooms(gt.copy(), sample_configs=sample_configs, out_zoom=combined_zoom) if gt else {}
        if output is not None:
            output_p = decode_zooms(output.copy(), sample_configs=sample_configs, out_zoom=combined_zoom)
        else:
            output_p = output_comp

        mask_p = {max_zoom: mask[max_zoom]} if mask is not None and max_zoom in mask else None
        self.log_tensor_plot(
            plot_types=["healpix_plot_zooms_var"],
            input=source_p,
            output=output_p,
            gt=target_p,
            mask=mask_p,
            sample_configs=sample_configs,
            emb=emb,
            plot_name=f"epoch_{current_epoch}_combined{plot_name}",
        )

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

        self.log_tensor_plot(
            plot_types=["regular_plot"],
            gt=gt_tensor,
            input=in_tensor,
            output=rec_tensor,
            target_coords=target_coords,
            in_coords=in_coords,
            plot_name=plot_name,
        )
