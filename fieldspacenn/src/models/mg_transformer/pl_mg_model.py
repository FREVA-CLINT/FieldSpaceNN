import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, List

import lightning.pytorch as pl
import torch
import torch.nn as nn
from pytorch_lightning.utilities import rank_zero_only
from ...modules.grids.grid_utils import decode_zooms
from ...utils.losses import MGMultiLoss, ReluPressureLevelScaler
from ...utils.schedulers import CosineWarmupScheduler
from ...utils.helpers import merge_sampling_dicts


class LightningMGModel(pl.LightningModule):
    def __init__(
        self,
        model: Any,
        lr_groups: Mapping[str, Mapping[str, Any]],
        lambda_loss_dict: Mapping[str, Any],
        data_variables: Optional[Mapping[str, Any]] = None,
        weight_decay: float = 0.0,
        lambda_loss_groups: List = [],
    ) -> None:
        """
        Initialize the Lightning wrapper for multi-grid transformer models.

        :param model: Multi-grid model instance.
        :param lr_groups: Optimizer parameter-group configuration, including optional
            per-group optimizer settings such as ``weight_decay`` and match rules
            for module classes or module names.
        :param lambda_loss_dict: Loss weighting dictionary.
        :param weight_decay: Fallback weight decay for groups that do not define one.
        :return: None.
        """
        
        super().__init__()

        self.model: Any = model
        self.lr_groups: Mapping[str, Mapping[str, Any]] = lr_groups  
        self.weight_decay: float = weight_decay
        self.loss_aggregation: str = str(lambda_loss_dict.get("aggregation", "mean")).lower()
        self.normalize_lambdas: bool = bool(lambda_loss_dict.get("normalize_lambdas", False))
        if self.loss_aggregation not in {"mean", "sum"}:
            raise ValueError(f"Unsupported loss aggregation '{self.loss_aggregation}'. Use 'mean' or 'sum'.")
        depth_loss_scaler_cfg = self._extract_depth_loss_scaler_config(lambda_loss_dict)
        self.depth_loss_scaler = ReluPressureLevelScaler(**depth_loss_scaler_cfg)
        self.loss_group_names, self.variable_loss_metadata = self._normalize_variable_loss_config(data_variables)
        
        self.save_hyperparameters(ignore=['model'])

        zooms_loss_dict = lambda_loss_dict.get("zooms",{})
        self.loss_zooms: MGMultiLoss = MGMultiLoss(
            zooms_loss_dict,
            grid_layers=model.grid_layers,
            normalize_lambdas=self.normalize_lambdas,
        )

        comp_loss_dict = lambda_loss_dict.get("composed",{})
        self.loss_composed: MGMultiLoss = MGMultiLoss(
            comp_loss_dict,
            grid_layers=model.grid_layers,
            normalize_lambdas=self.normalize_lambdas,
        )
        self.lambda_loss_groups = lambda_loss_groups

    @staticmethod
    def _dataset_applies_diff(dataset: Any) -> bool:
        """Return whether a dataset encodes finer zooms as residuals."""
        seen = set()
        while dataset is not None and id(dataset) not in seen:
            seen.add(id(dataset))
            if hasattr(dataset, "apply_diff"):
                return bool(dataset.apply_diff)
            dataset = getattr(dataset, "dataset", None)
        return True

    @staticmethod
    def _extract_depth_loss_scaler_config(
        lambda_loss_dict: Mapping[str, Any],
    ) -> Dict[str, float]:
        depth_loss_scaler = lambda_loss_dict.get("depth_loss_scaler", {})
        if not depth_loss_scaler:
            return {"minimum": 0.2, "slope": 0.001}

        if not isinstance(depth_loss_scaler, Mapping):
            raise ValueError(
                "`lambda_loss_dict.depth_loss_scaler` must be a mapping when provided."
            )

        minimum = float(depth_loss_scaler.get("minimum", 0.2))
        slope = float(depth_loss_scaler.get("slope", 0.001))
        return {"minimum": minimum, "slope": slope}

    @staticmethod
    def _normalize_variable_loss_config(
        data_variables: Optional[Mapping[str, Any]],
    ) -> tuple[List[str], Dict[str, Dict[str, Dict[str, Any]]]]:
        if not data_variables:
            return [], {}

        group_names: List[str] = []
        group_metadata: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for group_name, group_cfg in data_variables.items():
            if group_name == "embedding":
                continue

            group_names.append(str(group_name))
            group_metadata[str(group_name)] = {}

            if isinstance(group_cfg, Mapping):
                for var_name, var_cfg in group_cfg.items():
                    metadata: Dict[str, Any] = {"loss_lambda": 1.0}
                    if isinstance(var_cfg, Mapping):
                        metadata["variable_id"] = (
                            int(var_cfg["variable_id"]) if var_cfg.get("variable_id") is not None else None
                        )
                        if var_cfg.get("loss_lambda") is not None:
                            metadata["loss_lambda"] = float(var_cfg["loss_lambda"])
                    else:
                        metadata["variable_id"] = int(var_cfg) if var_cfg is not None else None
                    group_metadata[str(group_name)][str(var_name)] = metadata

        return group_names, group_metadata

    def _get_group_metadata(self, group_index: int) -> Dict[str, Dict[str, Any]]:
        if group_index >= len(self.loss_group_names):
            return {}
        return self.variable_loss_metadata.get(self.loss_group_names[group_index], {})

    def _find_variable_metadata(
        self,
        group_index: int,
        variable_name: Optional[str],
        variable_id: int,
    ) -> Dict[str, Any]:
        group_metadata = self._get_group_metadata(group_index)
        if variable_name is not None and variable_name in group_metadata:
            return group_metadata[variable_name]

        for metadata in group_metadata.values():
            if metadata.get("variable_id") == variable_id:
                return metadata

        return {"loss_lambda": 1.0, "variable_id": variable_id}

    def _build_group_variable_weight_map(
        self,
        group_index: int,
        group_output: Dict[int, torch.Tensor],
        emb_group: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        reference = list(group_output.values())[0]
        n_variables = reference.shape[1]
        depth_size = reference.shape[-2]
        if emb_group is None:
            return torch.ones((n_variables, depth_size), device=reference.device, dtype=reference.dtype)

        var_ids = emb_group.get("VariableEmbedder")
        if isinstance(var_ids, torch.Tensor):
            if var_ids.ndim > 1:
                var_ids_tensor = var_ids[0]
            else:
                var_ids_tensor = var_ids
        else:
            var_ids_tensor = torch.arange(n_variables, device=reference.device)

        var_ids_list = [int(var_id) for var_id in var_ids_tensor.tolist()]

        var_names = emb_group.get("variable_names_sampled", [])
        lambdas_level = emb_group.get("PressureLevelEmbedder")
        if isinstance(lambdas_level, torch.Tensor):
            if lambdas_level.ndim > 1:
                lambdas_level_tensor = lambdas_level[0]
            else:
                lambdas_level_tensor = lambdas_level
            lambdas_level_tensor = lambdas_level_tensor.to(device=reference.device, dtype=reference.dtype)
        else:
            lambdas_level_tensor = None

        if lambdas_level_tensor is not None and lambdas_level_tensor.dim()>1:
            if lambdas_level_tensor.numel() != depth_size:
                raise ValueError(
                    f"Depth values for group {group_index} have length {lambdas_level_tensor.numel()}, "
                    f"but runtime depth size is {depth_size}."
                )
            depth_weights = self.depth_loss_scaler(lambdas_level_tensor).to(dtype=reference.dtype)
        else:
            depth_weights = torch.ones(depth_size, device=reference.device, dtype=reference.dtype)

        base_weights = torch.tensor(
            [
                float(
                    self._find_variable_metadata(
                        group_index,
                        str(var_names[var_pos]) if var_pos < len(var_names) else None,
                        var_id,
                    ).get("loss_lambda", 1.0)
                )
                for var_pos, var_id in enumerate(var_ids_list)
            ],
            device=reference.device,
            dtype=reference.dtype,
        )

        return base_weights.unsqueeze(1) * depth_weights.unsqueeze(0)

    def _loss_normalizer(self, groups: Sequence[Dict[int, torch.Tensor]]) -> float:
        if self.loss_aggregation == "sum":
            return 1.0

        total = 0
        for group in groups:
            if not group:
                continue
            tensor = list(group.values())[0]
            total += int(tensor.shape[1] * tensor.shape[-2])
        return float(total if total > 0 else 1)

    def _group_lambda_normalizer(self, group_loss_inputs: Sequence[Dict[str, Any]]) -> float:
        if not self.normalize_lambdas or len(group_loss_inputs) == 0:
            return 1.0

        total_weight = 0.0
        total_entries = 0
        for group_input in group_loss_inputs:
            variable_weight_map = group_input["variable_weight_map"]
            total_weight += float(variable_weight_map.detach().sum().item()) * float(group_input["lambda_group"])
            total_entries += int(variable_weight_map.numel())

        if total_entries <= 0 or total_weight <= 0:
            return 1.0

        return total_weight / float(total_entries)

    @staticmethod
    def _merge_loss_dict(dest: Dict[str, float], src: Dict[str, float]) -> Dict[str, float]:
        for key, value in src.items():
            dest[key] = dest.get(key, 0.0) + float(value)
        return dest


    def forward(
        self,
        x_zooms_groups: Optional[Sequence[Dict[int, torch.Tensor]]] = None,
        mask_zooms_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        emb_groups: Optional[Sequence[Dict[str, Any]]] = None,
        sample_configs: Mapping[int, Dict[str, Any]] = {},
        out_zoom: Optional[int] = None,
        mask_zooms: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        emb: Optional[Sequence[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Forward pass through the wrapped multi-grid model.

        :param x_zooms_groups: List of per-group zoom mappings with tensors of shape
            ``(b, v, t, n, d, f)``.
        :param mask_zooms_groups: Optional list of mask mappings aligned with inputs.
        :param emb_groups: Optional list of embedding dictionaries aligned with inputs.
        :param sample_configs: Sampling configuration dictionary per zoom.
        :param out_zoom: Optional target zoom level to decode outputs into.
        :param mask_zooms: Optional mask mappings used when ``mask_zooms_groups`` is None.
        :param emb: Optional embeddings used when ``emb_groups`` is None.
        :param kwargs: Additional arguments forwarded to the model.
        :return: Model outputs as zoom-group mappings.
        """
        
        if x_zooms_groups is None:
            x_zooms_groups = []
        if isinstance(x_zooms_groups, dict):
            x_zooms_groups = [x_zooms_groups]

        if mask_zooms_groups is None:
            mask_zooms_groups = mask_zooms
        if emb_groups is None:
            emb_groups = emb

        return self.model( 
                x_zooms_groups=x_zooms_groups,
                mask_zooms_groups=mask_zooms_groups,
                emb_groups=emb_groups,
                sample_configs=sample_configs,
                out_zoom=out_zoom) 

    def get_losses(
        self,
        source_groups: Sequence[Dict[int, torch.Tensor]] | Dict[int, torch.Tensor],
        target_groups: Sequence[Dict[int, torch.Tensor]],
        sample_configs: Mapping[int, Dict[str, Any]] = {},
        sample_configs_target: Optional[Mapping[int, Dict[str, Any]]] = None,
        mask_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        emb_groups: Optional[Sequence[Dict[str, Any]]] = None,
        prefix: str = '',
        mask_zooms: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        emb: Optional[Sequence[Dict[str, Any]]] = None,
    ):
        """
        Compute losses for a batch.

        :param source_groups: Source zoom-group inputs with tensors of shape ``(b, v, t, n, d, f)``.
        :param target_groups: Target zoom-group inputs with tensors of shape ``(b, v, t, n, d, f)``.
        :param sample_configs: Sampling configuration dictionary per zoom.
        :param mask_groups: Optional mask groups aligned with inputs.
        :param emb_groups: Optional embedding groups aligned with inputs.
        :param prefix: Prefix for loss names.
        :param mask_zooms: Optional mask groups used when ``mask_groups`` is None.
        :param emb: Optional embeddings used when ``emb_groups`` is None.
        :return: Tuple of ``(total_loss, loss_dict, output_groups)``.
        """

        if mask_groups is None:
            mask_groups = mask_zooms
        if emb_groups is None:
            emb_groups = emb
        if sample_configs_target is None:
            sample_configs_target = sample_configs

        if isinstance(source_groups, dict):
            source_groups_list = [source_groups]
        else:
            source_groups_list = list(source_groups)

        output_groups = self(
            x_zooms_groups=[group.copy() for group in source_groups_list],
            mask_zooms_groups=mask_groups,
            emb_groups=emb_groups,
            sample_configs=sample_configs,
        )

        return self._compute_losses_from_output_groups(
            source_groups=source_groups_list,
            output_groups=output_groups,
            target_groups=target_groups,
            sample_configs=sample_configs,
            sample_configs_target=sample_configs_target,
            mask_groups=mask_groups,
            emb_groups=emb_groups,
            prefix=prefix,
        )

    def _compute_losses_from_output_groups(
        self,
        source_groups: Sequence[Dict[int, torch.Tensor]],
        output_groups: Sequence[Dict[int, torch.Tensor]],
        target_groups: Sequence[Dict[int, torch.Tensor]],
        sample_configs: Mapping[int, Dict[str, Any]],
        sample_configs_target: Mapping[int, Dict[str, Any]],
        mask_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        emb_groups: Sequence[Dict[str, Any]],
        prefix: str,
    ):
        loss_dict_total: Dict[str, float] = {}
        total_loss = 0
        group_loss_inputs = []

        lambda_groups = self.lambda_loss_groups if len(self.lambda_loss_groups) > 0 else [1.0] * len(source_groups)
        normalizer = self._loss_normalizer(target_groups)

        for group_index, (source, output, target, mask, emb, lambda_group) in enumerate(
            zip(source_groups, output_groups, target_groups, mask_groups, emb_groups, lambda_groups)
        ):
            variable_weight_map = self._build_group_variable_weight_map(group_index, target, emb)
            group_loss_inputs.append(
                {
                    "source": source,
                    "target": target,
                    "output": output,
                    "mask": mask,
                    "emb": emb,
                    "group_index": group_index,
                    "lambda_group": float(lambda_group),
                    "variable_weight_map": variable_weight_map,
                }
            )

        group_lambda_normalizer = self._group_lambda_normalizer(group_loss_inputs)

        for group_input in group_loss_inputs:
            effective_group_lambda = float(group_input["lambda_group"]) / float(group_lambda_normalizer)
            loss, loss_dict = self.loss_zooms(
                group_input["output"],
                group_input["target"],
                mask=group_input["mask"],
                sample_configs=sample_configs_target,
                prefix=f"{prefix}/",
                emb=group_input["emb"],
                variable_weight_map=group_input["variable_weight_map"],
                group_index=group_input["group_index"],
                group_lambda=effective_group_lambda,
                normalizer=normalizer,
            )
            total_loss += loss
            self._merge_loss_dict(loss_dict_total, loss_dict)

        if not self.loss_composed.has_elements:
            return total_loss, loss_dict_total, output_groups

        total_loss, loss_dict_total = self._add_composed_losses(
            group_loss_inputs=group_loss_inputs,
            sample_configs_target=sample_configs_target,
            prefix=prefix,
            total_loss=total_loss,
            loss_dict_total=loss_dict_total,
            normalizer=normalizer,
            group_lambda_normalizer=group_lambda_normalizer,
        )
        return total_loss, loss_dict_total, output_groups

    def _add_composed_losses(
        self,
        group_loss_inputs: Sequence[Dict[str, Any]],
        sample_configs_target: Mapping[int, Dict[str, Any]],
        prefix: str,
        total_loss: torch.Tensor | float,
        loss_dict_total: Dict[str, float],
        normalizer: float,
        group_lambda_normalizer: float,
    ) -> tuple[torch.Tensor | float, Dict[str, float]]:
        max_zoom = max((max(group["target"].keys()) for group in group_loss_inputs if group["target"]), default=None)
        if max_zoom is None:
            return total_loss, loss_dict_total

        for group_input in group_loss_inputs:
            if len(group_input["source"]) == 0:
                continue

            output_comp = decode_zooms(
                group_input["output"].copy(),
                sample_configs=sample_configs_target,
                out_zoom=max_zoom,
            )
            target_comp = decode_zooms(
                group_input["target"].copy(),
                sample_configs=sample_configs_target,
                out_zoom=max_zoom,
            )
            mask_comp = (
                decode_zooms(
                    group_input["mask"],
                    sample_configs=sample_configs_target,
                    out_zoom=max_zoom,
                )
                if group_input["mask"] is not None
                else None
            )

            loss, loss_dict = self.loss_composed(
                output_comp,
                target_comp,
                mask=mask_comp,
                sample_configs=sample_configs_target,
                prefix=f'{prefix}/composed_',
                emb=group_input["emb"],
                variable_weight_map=group_input["variable_weight_map"],
                group_index=group_input["group_index"],
                group_lambda=float(group_input["lambda_group"]) / float(group_lambda_normalizer),
                normalizer=normalizer,
            )
            total_loss += loss
            self._merge_loss_dict(loss_dict_total, loss_dict)

        return total_loss, loss_dict_total

    def training_step(
        self,
        batch: Tuple[Any, Any, Any, Any, Dict[int, torch.Tensor]],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Run one training step for the multi-grid model.

        :param batch: Tuple ``(source_groups, target_groups, mask_groups, emb_groups, patch_index_zooms)``
            with tensors shaped ``(b, v, t, n, d, f)`` per zoom.
        :param batch_idx: Index of the current batch.
        :return: Training loss tensor.
        """
        dataset = self.trainer.datamodule.dataset_train
        sample_configs = dataset.sampling_zooms_collate or dataset.sampling_zooms
        sample_configs_target = getattr(dataset, "sampling_zooms_target", sample_configs)
        source_groups, target_groups, mask_groups, emb_groups, patch_index_zooms = batch

        # Inject patch indices into the sampling configuration.
        sample_configs = merge_sampling_dicts(sample_configs, patch_index_zooms)
        sample_configs_target = merge_sampling_dicts(sample_configs_target, patch_index_zooms)

        loss, loss_dict, _ = self.get_losses(
            source_groups,
            target_groups,
            sample_configs,
            sample_configs_target=sample_configs_target,
            mask_groups=mask_groups,
            emb_groups=emb_groups,
            prefix='train',
        )
      
        self.log_dict({"train/total_loss": loss.item()}, prog_bar=True)
        self.log_dict(loss_dict, logger=True)
        
        return loss
    
    def validation_step(
        self,
        batch: Tuple[Any, Any, Any, Any, Dict[int, torch.Tensor]],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Run one validation step for the multi-grid model.

        :param batch: Tuple ``(source_groups, target_groups, mask_groups, emb_groups, patch_index_zooms)``
            with tensors shaped ``(b, v, t, n, d, f)`` per zoom.
        :param batch_idx: Index of the current batch.
        :return: Validation loss tensor.
        """
        dataset = self.trainer.datamodule.dataset_val
        sample_configs = dataset.sampling_zooms_collate or dataset.sampling_zooms
        sample_configs_target = getattr(dataset, "sampling_zooms_target", sample_configs)
        source_groups, target_groups, mask_groups, emb_groups, patch_index_zooms = batch

        max_zooms = [max(target.keys()) for target in target_groups if target]
        max_zoom = max(max_zooms) if max_zooms else max(self.model.in_zooms)

        # Inject patch indices into the sampling configuration.
        sample_configs = merge_sampling_dicts(sample_configs, patch_index_zooms)
        sample_configs_target = merge_sampling_dicts(sample_configs_target, patch_index_zooms)

   
        loss, loss_dict, output_groups = self.get_losses(
            [group.copy() for group in source_groups],
            target_groups,
            sample_configs=sample_configs, 
            sample_configs_target=sample_configs_target,
            mask_groups=mask_groups,
            emb_groups=emb_groups,
            prefix='val')
        
        self.log_dict({"validate/total_loss": loss.item()}, prog_bar=True)
        self.log_dict(loss_dict, logger=True)

        if batch_idx == 0 and rank_zero_only.rank==0:

            group_idx = next((idx for idx, group in enumerate(output_groups) if len(group) > 0), None)
            if group_idx is None:
                return loss

            output = output_groups[group_idx]
            source = source_groups[group_idx]
            target = target_groups[group_idx]
            mask = mask_groups[group_idx]
            emb = emb_groups[group_idx]

            self.logger.log_tensor_plot(
                plot_types=["healpix_plot_zooms_var"],
                input=source,
                output=output,
                gt=target,
                mask=mask,
                sample_configs=sample_configs,
                emb=emb,
                plot_name=f"epoch_{self.current_epoch}",
            )

            if self._dataset_applies_diff(dataset):
                output_comp = decode_zooms(
                    output.copy(), sample_configs=sample_configs_target, out_zoom=max_zoom
                )
                source_comp = decode_zooms(
                    source, sample_configs=sample_configs, out_zoom=max_zoom
                )
                target_comp = decode_zooms(
                    target, sample_configs=sample_configs_target, out_zoom=max_zoom
                )
                self.logger.log_tensor_plot(
                    plot_types=["healpix_plot_zooms_var"],
                    input=source_comp,
                    output=output_comp,
                    gt=target_comp,
                    mask={max_zoom: mask[max_zoom]} if mask is not None and max_zoom in mask else None,
                    sample_configs=sample_configs,
                    emb=emb,
                    plot_name=f"epoch_{self.current_epoch}_combined",
                )
            
        return loss


    def predict_step(
        self,
        batch: Tuple[Any, Any, Any, Any, Dict[int, torch.Tensor]],
        batch_idx: int,
    ) -> Dict[str, Any]:
        """
        Run one prediction step for the multi-grid model.

        :param batch: Tuple ``(source_groups, target_groups, mask_groups, emb_groups, patch_index_zooms)``
            with tensors shaped ``(b, v, t, n, d, f)`` per zoom.
        :param batch_idx: Index of the current batch.
        :return: Dictionary with outputs and masks.
        """
        sample_configs = self.trainer.predict_dataloaders.dataset.sampling_zooms_collate or self.trainer.predict_dataloaders.dataset.sampling_zooms
        source_groups, target_groups, mask_groups, emb_groups, patch_index_zooms = batch

        max_zoom = max(self.model.in_zooms)
        sample_configs = merge_sampling_dicts(sample_configs, patch_index_zooms)

        output = self([group.copy() for group in source_groups], sample_configs=sample_configs, mask_zooms=mask_groups, emb=emb_groups,
                           out_zoom=max_zoom)

        output = {
            'output': output,
            'mask': mask_groups,
        }
        return output

    def prepare_missing_zooms(
        self,
        x_zooms: Dict[int, torch.Tensor],
        sample_configs: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> Tuple[Dict[int, torch.Tensor], Optional[Dict[int, Dict[str, Any]]]]:
        """
        Fill missing zoom levels with zero tensors to match the maximum zoom shape.

        :param x_zooms: Mapping from zoom to tensor of shape ``(b, v, t, n, d, f)``.
        :param sample_configs: Optional sampling configuration dictionary per zoom.
        :return: Updated ``x_zooms`` and ``sample_configs``.
        """
        max_zoom = max(x_zooms.keys())
        for zoom in self.model.in_zooms:
            if zoom not in x_zooms.keys():
                x_zooms[zoom] = torch.zeros(1, 1, 1, 1, 1).expand(*x_zooms[max_zoom].shape[:3],
                                                                  int(x_zooms[max_zoom].shape[3] * 4**(zoom - max_zoom)),
                                                                  x_zooms[max_zoom].shape[4]).to(x_zooms[max_zoom].device)
                if sample_configs is not None:
                    sample_configs[zoom] = sample_configs[max_zoom]
        return x_zooms, sample_configs


    def configure_optimizers(self):
        """
        Build optimizer and cosine warmup scheduler.

        :return: Optimizer and scheduler configuration for Lightning.
        """
        if "default" not in self.lr_groups:
            raise ValueError("`lr_groups` must define a `default` group for unmatched parameters.")

        grouped_params = {group_name: [] for group_name in self.lr_groups}

        modules_by_path = dict(self.named_modules())

        def _normalize_match_keys(match_keys: Any) -> List[str]:
            if match_keys is None:
                return []
            if isinstance(match_keys, str):
                return [match_keys]
            return [str(match_key) for match_key in match_keys]

        def matches_group(
            module: nn.Module,
            module_path: str,
            parameter_name: str,
            group_name: str,
            group_cfg: Mapping[str, Any],
        ) -> bool:
            class_match_keys = _normalize_match_keys(group_cfg.get("matches", [group_name]))
            module_name_match_keys = _normalize_match_keys(group_cfg.get("module_name_matches"))
            parameter_name_match_keys = _normalize_match_keys(group_cfg.get("parameter_name_matches"))

            module_class_name = module.__class__.__name__
            class_match = any(match_key in module_class_name for match_key in class_match_keys)
            module_name_match = any(
                match_key == module_path.split(".")[-1] or match_key in module_path
                for match_key in module_name_match_keys
            )
            parameter_name_match = any(
                match_key == parameter_name.split(".")[-1] or match_key in parameter_name
                for match_key in parameter_name_match_keys
            )

            return class_match or module_name_match or parameter_name_match

        def get_group_name(parameter_name: str) -> str:
            module_path = parameter_name.rsplit(".", 1)[0] if "." in parameter_name else ""
            candidate_paths = [module_path]

            while candidate_paths[-1]:
                parent_path = candidate_paths[-1].rsplit(".", 1)[0] if "." in candidate_paths[-1] else ""
                candidate_paths.append(parent_path)

            for candidate_path in candidate_paths:
                module = modules_by_path[candidate_path]
                for group_name, group_cfg in self.lr_groups.items():
                    if matches_group(module, candidate_path, parameter_name, group_name, group_cfg):
                        return group_name

            return "default"

        for parameter_name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            grouped_params[get_group_name(parameter_name)].append(parameter)

        param_groups = []
        for group_name, group_cfg in self.lr_groups.items():
            group_options = {
                k: v
                for k, v in group_cfg.items()
                if k not in {"matches", "module_name_matches", "parameter_name_matches", "lr"}
            }
            group_options.setdefault("weight_decay", self.weight_decay)
            param_groups.append({
                "params": grouped_params[group_name],
                "lr": group_cfg["lr"],
                "name": group_name,
                **group_options,
            })

        optimizer = torch.optim.AdamW(param_groups)

        scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            max_iters=self.trainer.max_steps,
            iter_start=0
        )
        
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
    
    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """
        Hook executed before optimizer step (kept for debugging).

        :param optimizer: Optimizer instance about to step.
        :return: None.
        """
        #for debug only
        pass
        # Check for parameters with no gradients before optimizer.step()
       # print("Checking for parameters with None gradients before optimizer step:")
   #     for name, param in self.named_parameters():
   #         if param.grad is None:
   #             print(f"Parameter with no gradient: {name}")
