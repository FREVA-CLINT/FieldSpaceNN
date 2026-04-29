import copy
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..mg_transformer.mg_transformer import MG_Transformer


class MGDiffusionModel(nn.Module):
    """
    Multi-block diffusion model composed of sequential MG_Transformer modules.
    """

    def __init__(
        self,
        n_blocks: int,
        block_step_ranges: Sequence[Sequence[int]],
        mgrids: Sequence[Mapping[str, Any]],
        block_configs: Mapping[str, Any],
        in_zooms: Sequence[int],
        in_features: int = 1,
        n_groups_variables: Sequence[int] = (1,),
        inference_step_ranges: Optional[Sequence[Sequence[int]]] = None,
        pretrained_block_ckpt_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the sequential diffusion model with cloned MG_Transformer blocks.

        :param n_blocks: Number of sequential diffusion blocks.
        :param block_step_ranges: Training timestep ranges per block, inclusive, 0-based.
        :param mgrids: Multi-grid configuration used by each MG_Transformer block.
        :param block_configs: Block configuration mapping for MG_Transformer.
        :param in_zooms: Input zoom levels.
        :param in_features: Number of input features per variable.
        :param n_groups_variables: Number of variable groups for attention layers.
        :param inference_step_ranges: Optional inference ranges per block. If None, uses training ranges.
        :param pretrained_block_ckpt_path: Optional checkpoint path used to initialize the base block.
        :param kwargs: Additional arguments forwarded to MG_Transformer.
        :return: None.
        """
        super().__init__()

        if n_blocks <= 0:
            raise ValueError(f"`n_blocks` must be > 0, got {n_blocks}.")

        self.n_blocks: int = int(n_blocks)
        self.block_step_ranges: List[Tuple[int, int]] = self._validate_ranges(
            block_step_ranges, self.n_blocks, name="block_step_ranges"
        )
        self.inference_step_ranges: List[Tuple[int, int]] = (
            self._validate_ranges(inference_step_ranges, self.n_blocks, name="inference_step_ranges")
            if inference_step_ranges is not None
            else list(self.block_step_ranges)
        )

        base_block = MG_Transformer(
            mgrids=mgrids,
            block_configs=block_configs,
            in_zooms=in_zooms,
            in_features=in_features,
            n_groups_variables=n_groups_variables,
            **kwargs,
        )

        if pretrained_block_ckpt_path:
            self._load_pretrained_block_weights(base_block, pretrained_block_ckpt_path)

        self.blocks: nn.ModuleList = nn.ModuleList([copy.deepcopy(base_block) for _ in range(self.n_blocks)])

        # Expose common attributes expected by Lightning wrappers.
        self.in_zooms: Sequence[int] = self.blocks[0].in_zooms
        self.in_features: int = self.blocks[0].in_features
        self.grid_layers: nn.ModuleDict = self.blocks[0].grid_layers

    @staticmethod
    def _validate_ranges(
        ranges: Sequence[Sequence[int]],
        expected_len: int,
        name: str,
    ) -> List[Tuple[int, int]]:
        """
        Validate and normalize a per-block range configuration.

        :param ranges: Sequence of ranges ``[start, end]`` (inclusive).
        :param expected_len: Expected number of ranges.
        :param name: Field name used in validation errors.
        :return: Normalized list of integer tuples.
        """
        if ranges is None:
            raise ValueError(f"`{name}` cannot be None.")
        if len(ranges) != expected_len:
            raise ValueError(f"`{name}` length ({len(ranges)}) must equal n_blocks ({expected_len}).")

        normalized: List[Tuple[int, int]] = []
        for idx, step_range in enumerate(ranges):
            if len(step_range) != 2:
                raise ValueError(
                    f"`{name}[{idx}]` must have exactly two elements [start, end], got {step_range}."
                )
            start, end = int(step_range[0]), int(step_range[1])
            if start < 0 or end < 0:
                raise ValueError(
                    f"`{name}[{idx}]` values must be >= 0, got [{start}, {end}]."
                )
            if start > end:
                raise ValueError(
                    f"`{name}[{idx}]` requires start <= end, got [{start}, {end}]."
                )
            normalized.append((start, end))
        return normalized

    def validate_against_diffusion_steps(self, diffusion_steps: int) -> None:
        """
        Validate that configured ranges fit into ``[0, diffusion_steps-1]``.

        :param diffusion_steps: Total number of diffusion timesteps.
        :return: None.
        """
        max_step = int(diffusion_steps) - 1
        for name, ranges in (
            ("block_step_ranges", self.block_step_ranges),
            ("inference_step_ranges", self.inference_step_ranges),
        ):
            for idx, (start, end) in enumerate(ranges):
                if end > max_step:
                    raise ValueError(
                        f"`{name}[{idx}]` end={end} exceeds max diffusion step {max_step}."
                    )

    @staticmethod
    def _strip_prefix_if_present(state_dict: Mapping[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
        """
        Strip a prefix from all keys that start with it.

        :param state_dict: Source state_dict.
        :param prefix: Prefix to strip.
        :return: New state_dict with stripped keys where applicable.
        """
        return {
            (key[len(prefix):] if key.startswith(prefix) else key): value
            for key, value in state_dict.items()
        }

    @staticmethod
    def _build_prefix_candidates(state_dict: Mapping[str, torch.Tensor]) -> List[str]:
        """
        Build candidate prefixes that may need stripping to recover MG_Transformer keys.

        This covers checkpoints from:
        - plain MG_Transformer Lightning wrappers (``model.*``),
        - wrapped/distributed checkpoints (``module.*``),
        - MGDiffusionModel checkpoints with ``model.blocks.{i}.*``.

        :param state_dict: Source state_dict.
        :return: Ordered unique list of candidate prefixes.
        """
        prefixes: List[str] = [
            "",
            "model.",
            "module.",
            "module.model.",
            "model.model.",
            "model.blocks.0.",
            "blocks.0.",
            "module.model.blocks.0.",
            "model.model.blocks.0.",
        ]

        # Add discovered "model.blocks.{i}." prefixes from the checkpoint.
        block_prefixes = set()
        for key in state_dict.keys():
            match = re.search(r"(?:^|\.)(model\.)?blocks\.(\d+)\.", key)
            if match:
                idx = match.group(2)
                block_prefixes.add(f"blocks.{idx}.")
                block_prefixes.add(f"model.blocks.{idx}.")
                block_prefixes.add(f"module.model.blocks.{idx}.")
                block_prefixes.add(f"model.model.blocks.{idx}.")
            if ".Blocks." in key:
                # Recover everything before "Blocks." so stripped keys start at "Blocks."
                prefixes.append(key.split("Blocks.", 1)[0])

        prefixes.extend(sorted(block_prefixes))

        # Preserve order and deduplicate.
        seen = set()
        unique_prefixes: List[str] = []
        for prefix in prefixes:
            if prefix not in seen:
                seen.add(prefix)
                unique_prefixes.append(prefix)
        return unique_prefixes

    def _load_pretrained_block_weights(self, block: MG_Transformer, ckpt_path: str) -> None:
        """
        Load pretrained weights into a single MG_Transformer block.

        This supports checkpoints saved from Lightning modules where keys are often
        prefixed by ``model.`` or ``module.model.``.

        :param block: Block to initialize.
        :param ckpt_path: Path to checkpoint.
        :return: None.
        """
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)

        if not isinstance(state_dict, Mapping):
            raise ValueError(
                f"Checkpoint at `{ckpt_path}` does not contain a valid state_dict mapping."
            )

        target_keys = set(block.state_dict().keys())
        candidates: List[Dict[str, torch.Tensor]] = [
            self._strip_prefix_if_present(state_dict, prefix)
            for prefix in self._build_prefix_candidates(state_dict)
        ]

        best_candidate = None
        best_overlap = -1
        for candidate in candidates:
            overlap = len(target_keys.intersection(candidate.keys()))
            if overlap > best_overlap:
                best_overlap = overlap
                best_candidate = candidate

        if best_candidate is None or best_overlap <= 0:
            raise ValueError(
                f"No matching MG_Transformer keys found when loading `{ckpt_path}`."
            )

        block.load_state_dict(best_candidate, strict=False)

    def get_block(self, block_idx: int) -> MG_Transformer:
        """
        Return a block by index.

        :param block_idx: Block index.
        :return: MG_Transformer block.
        """
        return self.blocks[block_idx]

    def get_step_range(self, block_idx: int, inference: bool = False) -> Tuple[int, int]:
        """
        Return the timestep range for a block.

        :param block_idx: Block index.
        :param inference: If True, return inference range; otherwise training range.
        :return: Inclusive ``(start, end)`` range.
        """
        if inference:
            return self.inference_step_ranges[block_idx]
        return self.block_step_ranges[block_idx]

    def forward(
        self,
        x_zooms_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        mask_zooms_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        emb_groups: Optional[Sequence[Dict[str, Any]]] = None,
        sample_configs: Mapping[int, Any] = {},
        out_zoom: Optional[int] = None,
        return_all: bool = False,
    ):
        """
        Run the sequential block stack.

        :param x_zooms_groups: Input zoom-group mappings.
        :param mask_zooms_groups: Optional mask groups aligned with input groups.
        :param emb_groups: Optional embedding groups aligned with input groups.
        :param sample_configs: Sampling configuration dictionary per zoom.
        :param out_zoom: Optional target zoom level to decode outputs.
        :param return_all: Whether to return outputs from all intermediate blocks.
        :return: Final output groups, or all block outputs when ``return_all=True``.
        """
        if x_zooms_groups is None:
            x_zooms_groups = []

        current_groups = x_zooms_groups
        block_outputs = []

        for block in self.blocks:
            current_groups = block(
                x_zooms_groups=current_groups,
                mask_zooms_groups=mask_zooms_groups,
                emb_groups=emb_groups,
                sample_configs=sample_configs,
                out_zoom=out_zoom,
            )
            block_outputs.append(current_groups)

        if return_all:
            return block_outputs
        return current_groups
