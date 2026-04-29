import copy
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from pytorch_lightning.utilities import rank_zero_only

from .mg_diffusion_model import MGDiffusionModel
from ..mg_transformer.pl_mg_model import LightningMGModel
from ..mg_transformer.pl_mg_probabilistic import LightningProbabilisticModel
from ...modules.diffusion.mg_gaussian_diffusion import MGGaussianDiffusion
from ...modules.diffusion.mg_sampler import DDIMSampler, DDPMSampler
from ...modules.grids.grid_utils import decode_zooms
from ...utils.helpers import merge_sampling_dicts


class LightningMGDiffusionModel(LightningMGModel, LightningProbabilisticModel):
    """
    Lightning wrapper for sequential multi-block diffusion training and inference.
    """

    def __init__(
        self,
        model: MGDiffusionModel,
        gaussian_diffusion: MGGaussianDiffusion,
        lr_groups: Mapping[str, Mapping[str, Any]],
        lambda_loss_dict: Mapping[str, Any],
        weight_decay: float = 0.0,
        sampler: str = "ddpm",
        n_samples: int = 1,
        max_batchsize: int = -1,
        decode_zooms: bool = True,
        block_loss_mode: str = "final_only",
        block_loss_weights: Optional[Sequence[float]] = None,
    ) -> None:
        """
        Initialize the multi-block diffusion Lightning wrapper.

        :param model: Sequential diffusion model containing MG_Transformer blocks.
        :param gaussian_diffusion: Diffusion process helper.
        :param lr_groups: Optimizer parameter-group configuration.
        :param lambda_loss_dict: Loss weighting dictionary.
        :param weight_decay: Weight decay applied in the optimizer.
        :param sampler: Sampler name ("ddpm" or "ddim").
        :param n_samples: Number of posterior samples for probabilistic inference.
        :param max_batchsize: Optional cap on expanded prediction batch size.
        :param decode_zooms: Whether to decode prediction outputs to a single zoom.
        :param block_loss_mode: "final_only" or "all_blocks".
        :param block_loss_weights: Optional per-block weights used with "all_blocks".
        :return: None.
        """
        super().__init__(
            model=model,
            lr_groups=lr_groups,
            lambda_loss_dict=lambda_loss_dict,
            weight_decay=weight_decay,
        )

        self.gaussian_diffusion: MGGaussianDiffusion = gaussian_diffusion
        if sampler == "ddpm":
            self.sampler: DDPMSampler | DDIMSampler = DDPMSampler(self.gaussian_diffusion)
        else:
            self.sampler = DDIMSampler(self.gaussian_diffusion)

        self.n_samples: int = int(n_samples)
        self.max_batchsize: int = int(max_batchsize)
        self.decode_zooms: bool = bool(decode_zooms)

        if block_loss_mode not in {"final_only", "all_blocks"}:
            raise ValueError(
                f"`block_loss_mode` must be one of ['final_only', 'all_blocks'], got `{block_loss_mode}`."
            )
        self.block_loss_mode: str = block_loss_mode

        if block_loss_weights is None:
            self.block_loss_weights: Optional[List[float]] = (
                [1.0 / float(self.model.n_blocks)] * self.model.n_blocks
                if self.block_loss_mode == "all_blocks"
                else None
            )
        else:
            if len(block_loss_weights) != self.model.n_blocks:
                raise ValueError(
                    "`block_loss_weights` length must match model.n_blocks. "
                    f"Got {len(block_loss_weights)} vs {self.model.n_blocks}."
                )
            self.block_loss_weights = [float(weight) for weight in block_loss_weights]

        self.model.validate_against_diffusion_steps(self.gaussian_diffusion.diffusion_steps)

    def forward(
        self,
        x_zooms_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        sample_configs: Mapping[int, Any] = {},
        mask_zooms_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        emb_groups: Optional[Sequence[Dict[str, Any]]] = None,
        out_zoom: Optional[int] = None,
        return_all: bool = False,
    ):
        """
        Forward call into the sequential diffusion model.

        :param x_zooms_groups: Input zoom-group mappings.
        :param sample_configs: Sampling configuration dictionary per zoom.
        :param mask_zooms_groups: Optional mask groups aligned with inputs.
        :param emb_groups: Optional embedding groups aligned with inputs.
        :param out_zoom: Optional target zoom level for decode.
        :param return_all: Whether to return all intermediate block outputs.
        :return: Final model outputs, or all intermediate outputs.
        """
        return self.model(
            x_zooms_groups=x_zooms_groups,
            mask_zooms_groups=mask_zooms_groups,
            emb_groups=emb_groups,
            sample_configs=sample_configs,
            out_zoom=out_zoom,
            return_all=return_all,
        )

    @staticmethod
    def _copy_groups(
        groups: Sequence[Optional[Dict[int, torch.Tensor]]],
    ) -> List[Optional[Dict[int, torch.Tensor]]]:
        """
        Create a shallow copy of grouped zoom dictionaries.

        :param groups: Grouped zoom mappings.
        :return: Copied list of grouped zoom mappings.
        """
        return [group.copy() if group else None for group in groups]

    @staticmethod
    def _get_first_valid_group(
        groups: Sequence[Optional[Dict[int, torch.Tensor]]],
    ) -> Optional[Dict[int, torch.Tensor]]:
        """
        Return the first non-empty group from a list.

        :param groups: Grouped zoom mappings.
        :return: First valid group or None.
        """
        return next((group for group in groups if group), None)

    def _get_batch_size_and_device(
        self,
        groups: Sequence[Optional[Dict[int, torch.Tensor]]],
    ) -> Tuple[int, torch.device]:
        """
        Infer batch size and device from grouped zoom tensors.

        :param groups: Grouped zoom mappings.
        :return: Tuple of (batch_size, device).
        """
        first_valid_group = self._get_first_valid_group(groups)
        if not first_valid_group:
            return 0, self.device
        max_zoom = max(first_valid_group.keys())
        tensor = first_valid_group[max_zoom]
        return int(tensor.shape[0]), tensor.device

    def _get_sample_configs(self, stage: str) -> Mapping[int, Any]:
        """
        Fetch dataset sampling configuration by stage.

        :param stage: "train", "val", or "predict".
        :return: Sampling configuration dictionary.
        """
        if stage == "predict":
            dataset = self.trainer.predict_dataloaders.dataset
        else:
            dataset = self.trainer.val_dataloaders.dataset
        return dataset.sampling_zooms_collate or dataset.sampling_zooms

    def _sample_block_diffusion_steps(
        self,
        block_idx: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Uniformly sample diffusion steps within a block-specific range.

        :param block_idx: Block index.
        :param batch_size: Batch size.
        :param device: Device for sampled indices.
        :return: Timestep tensor of shape ``(batch_size,)``.
        """
        start, end = self.model.get_step_range(block_idx, inference=False)
        return torch.randint(start, end + 1, size=(batch_size,), device=device, dtype=torch.long)

    @staticmethod
    def _extract_training_losses(
        diffusion_outputs: Sequence[Tuple[Any, Any, Any]],
    ) -> Tuple[List[Optional[Dict[int, torch.Tensor]]], List[Optional[Dict[int, torch.Tensor]]], List[Optional[Dict[int, torch.Tensor]]]]:
        """
        Split diffusion output tuples into target, prediction, and pred_xstart groups.

        :param diffusion_outputs: Output from ``MGGaussianDiffusion.training_losses``.
        :return: Tuple of (targets, predictions, pred_xstart).
        """
        target_groups: List[Optional[Dict[int, torch.Tensor]]] = []
        output_groups: List[Optional[Dict[int, torch.Tensor]]] = []
        pred_xstart_groups: List[Optional[Dict[int, torch.Tensor]]] = []

        for group_output in diffusion_outputs:
            if group_output is None or len(group_output) < 2:
                target_groups.append(None)
                output_groups.append(None)
                pred_xstart_groups.append(None)
                continue

            target_groups.append(group_output[0])
            output_groups.append(group_output[1])
            pred_xstart_groups.append(group_output[2] if len(group_output) > 2 else None)

        return target_groups, output_groups, pred_xstart_groups

    def _compute_losses_from_diffusion_outputs(
        self,
        source_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        output_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        target_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        sample_configs: Mapping[int, Dict[str, Any]],
        mask_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]],
        emb_groups: Optional[Sequence[Optional[Dict[str, Any]]]],
        prefix: str,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute supervised losses from diffusion targets and outputs.

        :param source_groups: Source group inputs used for weighting.
        :param output_groups: Model output groups.
        :param target_groups: Target groups from diffusion objective.
        :param sample_configs: Sampling configuration dictionary per zoom.
        :param mask_groups: Optional mask groups.
        :param emb_groups: Optional embedding groups.
        :param prefix: Prefix used in logged loss names.
        :return: Tuple of (total_loss, loss_dict).
        """
        mask_groups = list(mask_groups) if mask_groups is not None else [None] * len(source_groups)
        emb_groups = list(emb_groups) if emb_groups is not None else [None] * len(source_groups)

        valid_indices = [
            idx
            for idx, (source, output, target) in enumerate(zip(source_groups, output_groups, target_groups))
            if source and output and target
        ]

        if not valid_indices:
            return torch.tensor(0.0, device=self.device), {}

        first_valid_source = source_groups[valid_indices[0]]
        assert first_valid_source is not None
        device = next(iter(first_valid_source.values())).device
        total_loss = torch.tensor(0.0, device=device)
        loss_dict_total: Dict[str, torch.Tensor] = {}

        if len(self.lambda_loss_groups) not in {0, len(source_groups)}:
            raise ValueError(
                "`lambda_loss_groups` must be empty or match the number of groups. "
                f"Got {len(self.lambda_loss_groups)} and {len(source_groups)}."
            )
        lambda_groups = (
            list(self.lambda_loss_groups)
            if len(self.lambda_loss_groups) > 0
            else [1.0] * len(source_groups)
        )

        group_var_counts = []
        for idx in valid_indices:
            source = source_groups[idx]
            assert source is not None
            group_var_counts.append(float(next(iter(source.values())).shape[1]))
        group_weights = torch.tensor(group_var_counts, device=device, dtype=torch.float32)
        group_weights = group_weights / group_weights.sum()

        for local_idx, idx in enumerate(valid_indices):
            source = source_groups[idx]
            output = output_groups[idx]
            target = target_groups[idx]
            mask = mask_groups[idx]
            emb = emb_groups[idx]
            lambda_group = float(lambda_groups[idx])
            weight_group = group_weights[local_idx]

            assert source is not None and output is not None and target is not None

            loss, loss_dict = self.loss_zooms(
                output,
                target,
                mask=mask,
                sample_configs=sample_configs,
                prefix=f"{prefix}/",
                emb=emb,
            )
            total_loss = total_loss + loss * (lambda_group * weight_group)
            loss_dict_total.update(loss_dict)

        if self.loss_composed.has_elements:
            max_zooms = [max(target.keys()) for target in target_groups if target]
            if max_zooms:
                max_zoom = max(max_zooms)
                for local_idx, idx in enumerate(valid_indices):
                    source = source_groups[idx]
                    output = output_groups[idx]
                    target = target_groups[idx]
                    mask = mask_groups[idx]
                    emb = emb_groups[idx]
                    lambda_group = float(lambda_groups[idx])
                    weight_group = group_weights[local_idx]

                    if not source or not output or not target:
                        continue

                    output_comp = decode_zooms(output.copy(), sample_configs=sample_configs, out_zoom=max_zoom)
                    target_comp = decode_zooms(target.copy(), sample_configs=sample_configs, out_zoom=max_zoom)

                    loss, loss_dict = self.loss_composed(
                        output_comp,
                        target_comp,
                        mask=mask,
                        sample_configs=sample_configs,
                        prefix=f"{prefix}/composed_",
                        emb=emb,
                    )
                    total_loss = total_loss + loss * (lambda_group * weight_group)
                    loss_dict_total.update(loss_dict)

        return total_loss, loss_dict_total

    def _run_single_block_training_loss(
        self,
        block_idx: int,
        input_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        sample_configs: Mapping[int, Dict[str, Any]],
        mask_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        emb_groups: Sequence[Optional[Dict[str, Any]]],
        prefix: str,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], List[Optional[Dict[int, torch.Tensor]]]]:
        """
        Run one block's diffusion objective and compute its supervised loss.

        :param block_idx: Block index.
        :param input_groups: Input groups for this block.
        :param sample_configs: Sampling configuration dictionary per zoom.
        :param mask_groups: Mask groups aligned with input groups.
        :param emb_groups: Embedding groups aligned with input groups.
        :param prefix: Prefix used for block-specific loss logging.
        :return: Tuple of (block_loss, block_loss_dict, pred_xstart_groups).
        """
        batch_size, device = self._get_batch_size_and_device(input_groups)
        if batch_size == 0:
            return torch.tensor(0.0, device=self.device), {}, [None] * len(input_groups)

        diffusion_steps = self._sample_block_diffusion_steps(block_idx, batch_size, device)
        diffusion_outputs = self.gaussian_diffusion.training_losses(
            self.model.get_block(block_idx),
            input_groups,
            diffusion_steps,
            mask_groups=mask_groups,
            emb_groups=emb_groups,
            create_pred_xstart=True,
            sample_configs=sample_configs,
        )

        target_groups, output_groups, pred_xstart_groups = self._extract_training_losses(diffusion_outputs)
        block_loss, block_loss_dict = self._compute_losses_from_diffusion_outputs(
            source_groups=input_groups,
            output_groups=output_groups,
            target_groups=target_groups,
            sample_configs=sample_configs,
            mask_groups=mask_groups,
            emb_groups=emb_groups,
            prefix=f"{prefix}/block_{block_idx}",
        )
        return block_loss, block_loss_dict, pred_xstart_groups

    def _aggregate_block_losses(self, block_losses: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        Aggregate per-block losses according to the configured block loss mode.

        :param block_losses: Ordered list of block losses.
        :return: Aggregated total loss.
        """
        if len(block_losses) == 0:
            return torch.tensor(0.0, device=self.device)

        if self.block_loss_mode == "final_only":
            return block_losses[-1]

        assert self.block_loss_weights is not None
        weights = torch.tensor(
            self.block_loss_weights,
            dtype=block_losses[0].dtype,
            device=block_losses[0].device,
        )
        return sum(weight * loss for weight, loss in zip(weights, block_losses))

    def training_step(
        self,
        batch: Tuple[Any, Any, Any, Any, Dict[int, torch.Tensor]],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Run one training step with sequential block chaining.

        :param batch: Tuple ``(source_groups, target_groups, mask_groups, emb_groups, patch_index_zooms)``.
        :param batch_idx: Index of the current batch.
        :return: Training loss tensor.
        """
        sample_configs = self._get_sample_configs(stage="train")
        source_groups, target_groups, mask_groups, emb_groups, patch_index_zooms = batch
        sample_configs = merge_sampling_dicts(sample_configs, patch_index_zooms)

        mask_groups = mask_groups if mask_groups is not None else [None] * len(target_groups)
        emb_groups = emb_groups if emb_groups is not None else [None] * len(target_groups)

        current_groups = self._copy_groups(target_groups)
        block_losses: List[torch.Tensor] = []
        total_loss_dict: Dict[str, torch.Tensor] = {}

        for block_idx in range(self.model.n_blocks):
            block_loss, block_loss_dict, pred_xstart_groups = self._run_single_block_training_loss(
                block_idx=block_idx,
                input_groups=current_groups,
                sample_configs=sample_configs,
                mask_groups=mask_groups,
                emb_groups=emb_groups,
                prefix="train",
            )
            block_losses.append(block_loss)
            total_loss_dict.update(block_loss_dict)
            current_groups = self._copy_groups(pred_xstart_groups)

        total_loss = self._aggregate_block_losses(block_losses)

        self.log_dict({"train/total_loss": total_loss.item()}, prog_bar=True)
        self.log_dict(total_loss_dict, logger=True)
        return total_loss

    def validation_step(
        self,
        batch: Tuple[Any, Any, Any, Any, Dict[int, torch.Tensor]],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Run one validation step with sequential block chaining.

        :param batch: Tuple ``(source_groups, target_groups, mask_groups, emb_groups, patch_index_zooms)``.
        :param batch_idx: Index of the current batch.
        :return: Validation loss tensor.
        """
        sample_configs = self._get_sample_configs(stage="val")
        source_groups, target_groups, mask_groups, emb_groups, patch_index_zooms = batch
        sample_configs = merge_sampling_dicts(sample_configs, patch_index_zooms)

        mask_groups = mask_groups if mask_groups is not None else [None] * len(target_groups)
        emb_groups = emb_groups if emb_groups is not None else [None] * len(target_groups)

        current_groups = self._copy_groups(target_groups)
        block_losses: List[torch.Tensor] = []
        total_loss_dict: Dict[str, torch.Tensor] = {}

        for block_idx in range(self.model.n_blocks):
            block_loss, block_loss_dict, pred_xstart_groups = self._run_single_block_training_loss(
                block_idx=block_idx,
                input_groups=current_groups,
                sample_configs=sample_configs,
                mask_groups=mask_groups,
                emb_groups=emb_groups,
                prefix="val",
            )
            block_losses.append(block_loss)
            total_loss_dict.update(block_loss_dict)
            current_groups = self._copy_groups(pred_xstart_groups)

        total_loss = self._aggregate_block_losses(block_losses)
        self.log_dict({"val/total_loss": total_loss.item()}, prog_bar=True)
        self.log_dict(total_loss_dict, logger=True)

        if batch_idx == 0 and rank_zero_only.rank == 0:
            self.log_healpix_tensor_plot(
                source_groups=source_groups,
                target_groups=target_groups,
                mask_groups=mask_groups,
                emb_groups=emb_groups,
                patch_index_zooms=patch_index_zooms,
            )

        return total_loss

    @staticmethod
    def _slice_batch_item(value: Any, index: int = 0) -> Any:
        """
        Slice the first batch element from nested tensor/dict structures.

        :param value: Nested value that may contain tensors and dicts.
        :param index: Batch index to slice.
        :return: Sliced nested value.
        """
        if isinstance(value, torch.Tensor):
            if value.ndim > 0:
                return value[index:index + 1]
            return value
        if isinstance(value, dict):
            return {k: LightningMGDiffusionModel._slice_batch_item(v, index=index) for k, v in value.items()}
        return value

    def log_healpix_tensor_plot(
        self,
        source_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        target_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        mask_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        emb_groups: Sequence[Optional[Dict[str, Any]]],
        patch_index_zooms: Dict[int, torch.Tensor],
    ) -> None:
        """
        Log diffusion denoising snapshots for selected timesteps on one validation sample.

        This mirrors the single-block diffusion visualization flow and adapts it to
        sequential multi-block denoising by chaining each block's ``pred_xstart`` into
        the next block.

        :param source_groups: Source zoom-group inputs.
        :param target_groups: Target zoom-group inputs.
        :param mask_groups: Mask groups aligned with inputs.
        :param emb_groups: Embedding groups aligned with inputs.
        :param patch_index_zooms: Patch indices per zoom.
        :return: None.
        """
        group_idx = next(
            (
                idx
                for idx, (source_group, target_group) in enumerate(zip(source_groups, target_groups))
                if source_group and target_group
            ),
            None,
        )
        if group_idx is None:
            return

        source_group = source_groups[group_idx]
        target_group = target_groups[group_idx]
        mask_group = mask_groups[group_idx] if mask_groups and group_idx < len(mask_groups) else None
        emb_group = emb_groups[group_idx] if emb_groups and group_idx < len(emb_groups) else None
        if source_group is None or target_group is None:
            return

        max_zooms = [max(group.keys()) for group in target_groups if group]
        max_zoom = max(max_zooms) if max_zooms else max(self.model.in_zooms)

        device = source_group[max(source_group.keys())].device
        ts = torch.tensor(
            [(self.gaussian_diffusion.diffusion_steps // 4) * (x + 1) - 1 for x in range(4)],
            device=device,
        )

        for t in ts:
            source_p = {zoom: source_group[zoom][0:1] for zoom in source_group.keys()}
            target_p = {zoom: target_group[zoom][0:1] for zoom in target_group.keys()}
            mask_p = (
                {zoom: mask_group[zoom][0:1] for zoom in mask_group.keys()}
                if mask_group
                else None
            )
            emb_p = self._slice_batch_item(emb_group, index=0) if emb_group else None

            patch_index_zooms_p = {zoom: patch_index_zooms[zoom][0:1] for zoom in patch_index_zooms.keys()}
            sample_configs_p = merge_sampling_dicts(
                copy.deepcopy(self._get_sample_configs(stage="val")),
                patch_index_zooms_p,
            )

            current_groups: List[Optional[Dict[int, torch.Tensor]]] = [target_p.copy()]
            mask_groups_p = [mask_p.copy()] if mask_p else [None]
            emb_groups_p = [emb_p] if emb_p else [None]

            for block_idx in range(self.model.n_blocks):
                pred_xstart_outputs = self.gaussian_diffusion.training_losses(
                    self.model.get_block(block_idx),
                    current_groups,
                    torch.stack([t]),
                    mask_groups=mask_groups_p,
                    emb_groups=emb_groups_p,
                    create_pred_xstart=True,
                    sample_configs=sample_configs_p,
                )

                _, _, pred_xstart_groups = self._extract_training_losses(pred_xstart_outputs)
                current_groups = self._copy_groups(pred_xstart_groups)

            pred_xstart_group = current_groups[0] if current_groups else None
            if not pred_xstart_group:
                continue

            if self.decode_zooms:
                pred_xstart_comp = decode_zooms(
                    pred_xstart_group.copy(),
                    sample_configs=sample_configs_p,
                    out_zoom=max_zoom,
                )
            else:
                pred_xstart_comp = {max_zoom: pred_xstart_group[max_zoom]}

            self.logger.log_healpix_tensor_plot(
                source_p,
                pred_xstart_group,
                target_p,
                mask_p,
                sample_configs_p,
                emb_p,
                max_zoom,
                self.current_epoch,
                output_comp=pred_xstart_comp,
                plot_name=f"_{t.item()}",
            )

    def predict_step(
        self,
        batch: Tuple[Any, Any, Any, Any, Dict[int, torch.Tensor]],
        batch_idx: int,
    ) -> Dict[str, Any]:
        """
        Run prediction using the probabilistic parent implementation.

        :param batch: Prediction batch tuple.
        :param batch_idx: Index of the current batch.
        :return: Prediction output dictionary.
        """
        return LightningProbabilisticModel.predict_step(self, batch, batch_idx)

    def _sample_block_range(
        self,
        block_idx: int,
        input_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        mask_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        emb_groups: Sequence[Optional[Dict[str, Any]]],
        sample_configs: Mapping[int, Dict[str, Any]],
        initialize_from_noise: bool = True,
    ) -> List[Optional[Dict[int, torch.Tensor]]]:
        """
        Run reverse diffusion for one block over its configured inference range.

        :param block_idx: Block index.
        :param input_groups: Conditioning groups for this block.
        :param mask_groups: Optional masks aligned with inputs.
        :param emb_groups: Embeddings aligned with inputs.
        :param sample_configs: Sampling configuration dictionary per zoom.
        :param initialize_from_noise: If True, initialize this block from Gaussian
            noise. If False, start directly from ``input_groups``.
        :return: Final sampled groups for this block.
        """
        if initialize_from_noise:
            x_t_groups = [
                self.gaussian_diffusion.generate_noise(group) if group else None
                for group in input_groups
            ]
        else:
            x_t_groups = self._copy_groups(input_groups)

        first_valid_group = self._get_first_valid_group(x_t_groups)
        if not first_valid_group:
            return list(x_t_groups)

        device = next(iter(first_valid_group.values())).device
        batch_size = next(iter(first_valid_group.values())).shape[0]

        for i, (mask_zooms, input_zooms, x_t_zooms) in enumerate(zip(mask_groups, input_groups, x_t_groups)):
            if mask_zooms and input_zooms and x_t_zooms:
                x_t_groups[i] = {
                    int(zoom): torch.where(~mask_zooms[zoom], input_zooms[zoom], x_t_zooms[zoom])
                    for zoom in x_t_zooms.keys()
                }

        start_step, end_step = self.model.get_step_range(block_idx, inference=True)
        step_indices = range(end_step, start_step - 1, -1)

        for step in step_indices:
            diffusion_steps = torch.full((batch_size,), step, device=device, dtype=torch.long)
            out = self.sampler.sample(
                self.model.get_block(block_idx),
                x_t_groups,
                diffusion_steps,
                mask_groups=mask_groups,
                sample_configs=sample_configs,
                emb_groups=emb_groups,
            )

            if mask_groups is not None:
                for j, (mask_zooms, input_zooms, sample_zooms) in enumerate(
                    zip(mask_groups, input_groups, out["sample"])
                ):
                    if mask_zooms and input_zooms and sample_zooms:
                        out["sample"][j] = {
                            int(zoom): torch.where(~mask_zooms[zoom], input_zooms[zoom], sample_zooms[zoom])
                            for zoom in input_zooms.keys()
                        }

            x_t_groups = out["sample"]

        return list(x_t_groups)

    def _predict_step(
        self,
        source_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        target_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        patch_index_zooms: Dict[int, torch.Tensor],
        mask_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        emb_groups: Sequence[Optional[Dict[str, Any]]],
    ) -> List[Optional[Dict[int, torch.Tensor]]]:
        """
        Internal prediction step with sequential block-wise diffusion sampling.

        :param source_groups: Source zoom-group inputs.
        :param target_groups: Target zoom-group inputs.
        :param patch_index_zooms: Patch indices per zoom.
        :param mask_groups: Mask groups aligned with inputs.
        :param emb_groups: Embedding groups aligned with inputs.
        :return: Output zoom-group mappings.
        """
        sample_configs = self._get_sample_configs(stage="predict")
        sample_configs = merge_sampling_dicts(sample_configs, patch_index_zooms)

        mask_groups = mask_groups if mask_groups is not None else [None] * len(source_groups)
        emb_groups = emb_groups if emb_groups is not None else [None] * len(source_groups)

        current_groups = self._copy_groups(source_groups)

        for block_idx in range(self.model.n_blocks):
            current_groups = self._sample_block_range(
                block_idx=block_idx,
                input_groups=current_groups,
                mask_groups=mask_groups,
                emb_groups=emb_groups,
                sample_configs=sample_configs,
                initialize_from_noise=(block_idx == 0),
            )

        if not self.decode_zooms:
            return current_groups

        max_zoom = None
        first_target_group = self._get_first_valid_group(target_groups)
        if first_target_group:
            max_zoom = max(first_target_group.keys())
        elif len(self.model.in_zooms) > 0:
            max_zoom = max(self.model.in_zooms)

        if max_zoom is None:
            return current_groups

        decoded_outputs: List[Optional[Dict[int, torch.Tensor]]] = []
        for group in current_groups:
            if group:
                decoded_outputs.append(
                    decode_zooms(group.copy(), sample_configs=sample_configs, out_zoom=max_zoom)
                )
            else:
                decoded_outputs.append(None)
        return decoded_outputs
