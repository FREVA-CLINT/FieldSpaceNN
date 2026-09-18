import copy
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..mg_transformer.mg_transformer import MG_Transformer


class MGFlowMatchingModel(nn.Module):
    """
    Multi-block flow-matching model composed of sequential MG_Transformer modules.
    """

    def __init__(
        self,
        n_blocks: int,
        block_time_ranges: Sequence[Sequence[float]],
        mgrids: Sequence[Mapping[str, Any]],
        block_configs: Mapping[str, Any],
        in_zooms: Sequence[int],
        in_features: int = 1,
        n_groups_variables: Sequence[int] = (1,),
        inference_time_ranges: Optional[Sequence[Sequence[float]]] = None,
        pretrained_block_ckpt_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the sequential flow-matching model with cloned MG_Transformer blocks.
        """
        super().__init__()

        if n_blocks <= 0:
            raise ValueError(f"`n_blocks` must be > 0, got {n_blocks}.")

        self.n_blocks: int = int(n_blocks)
        self.block_time_ranges: List[Tuple[float, float]] = self._validate_ranges(
            block_time_ranges, self.n_blocks, name="block_time_ranges"
        )
        self.inference_time_ranges: List[Tuple[float, float]] = (
            self._validate_ranges(inference_time_ranges, self.n_blocks, name="inference_time_ranges")
            if inference_time_ranges is not None
            else list(self.block_time_ranges)
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

        self.in_zooms: Sequence[int] = self.blocks[0].in_zooms
        self.in_features: int = self.blocks[0].in_features
        self.grid_layers: nn.ModuleDict = self.blocks[0].grid_layers

    @staticmethod
    def _validate_ranges(
        ranges: Sequence[Sequence[float]],
        expected_len: int,
        name: str,
    ) -> List[Tuple[float, float]]:
        """
        Validate and normalize per-block continuous time ranges.
        """
        if ranges is None:
            raise ValueError(f"`{name}` cannot be None.")
        if len(ranges) != expected_len:
            raise ValueError(f"`{name}` length ({len(ranges)}) must equal n_blocks ({expected_len}).")

        normalized: List[Tuple[float, float]] = []
        for idx, time_range in enumerate(ranges):
            if len(time_range) != 2:
                raise ValueError(f"`{name}[{idx}]` must have exactly two elements [start, end].")
            start, end = float(time_range[0]), float(time_range[1])
            if start < 0.0 or end > 1.0:
                raise ValueError(f"`{name}[{idx}]` values must be in [0, 1], got [{start}, {end}].")
            if start > end:
                raise ValueError(f"`{name}[{idx}]` requires start <= end, got [{start}, {end}].")
            normalized.append((start, end))
        return normalized

    @staticmethod
    def _strip_prefix_if_present(state_dict: Mapping[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
        return {
            (key[len(prefix):] if key.startswith(prefix) else key): value
            for key, value in state_dict.items()
        }

    @staticmethod
    def _build_prefix_candidates(state_dict: Mapping[str, torch.Tensor]) -> List[str]:
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
                prefixes.append(key.split("Blocks.", 1)[0])

        prefixes.extend(sorted(block_prefixes))

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
        """
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)

        if not isinstance(state_dict, Mapping):
            raise ValueError(f"Checkpoint at `{ckpt_path}` does not contain a valid state_dict mapping.")

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
            raise ValueError(f"No matching MG_Transformer keys found when loading `{ckpt_path}`.")

        block.load_state_dict(best_candidate, strict=False)

    def get_block(self, block_idx: int) -> MG_Transformer:
        """
        Return a block by index.
        """
        return self.blocks[block_idx]

    def get_time_range(self, block_idx: int, inference: bool = False) -> Tuple[float, float]:
        """
        Return the continuous time range for a block.
        """
        if inference:
            return self.inference_time_ranges[block_idx]
        return self.block_time_ranges[block_idx]

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
