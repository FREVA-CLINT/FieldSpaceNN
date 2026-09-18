from collections.abc import Sequence as SequenceCollection
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import torch
import torch.nn as nn

from ..mg_transformer.mg_base_model import MG_base_model
from ..mg_transformer.block_wrap_operations import (
    BlockWrapConfig,
    BlockWrapContext,
    BlockWrapOperation,
    create_block_wrap_operation,
)
from ...modules.field_space.healpix_convolution import MultiZoomHealpixConvBase


class HealpixUNet(MG_base_model):
    """U-Net for nested HEALPix tensors with feature-wise skip concatenation."""

    def __init__(
        self,
        mgrids: Sequence[Mapping[str, Any]],
        input_zoom: int,
        bottleneck_zoom: int,
        features: Sequence[int],
        in_features: int = 1,
        out_features: int = 1,
        n_groups_variables: Sequence[int] = (1,),
        use_neighborhood: bool = True,
        norm: str = "group",
        act: str = "silu",
        residual: bool = True,
        num_groups: Union[int, Sequence[int]] = 8,
        blocks_per_level: Union[int, Sequence[int]] = 1,
        eps: float = 1e-5,
        rank_space: Union[Optional[int], Sequence[Optional[int]]] = None,
        rank_time: Union[Optional[int], Sequence[Optional[int]]] = None,
        rank_depth: Union[Optional[int], Sequence[Optional[int]]] = None,
        fac_mode: str = "Tucker",
        layer_confs: Optional[
            Union[Mapping[str, Any], Sequence[Mapping[str, Any]]]
        ] = None,
        add_refined_input: Optional[BlockWrapConfig] = None,
        **kwargs: Any,
    ) -> None:
        """
        Build a HEALPix U-Net over every zoom from input to bottleneck.

        ``features`` is ordered from ``input_zoom`` down to ``bottleneck_zoom``.
        ``blocks_per_level`` follows the same order. Its first block is the
        stem or downsampling block in the encoder and the skip-fusion block in
        the decoder; values greater than one add same-zoom refinement blocks.
        Every encoder output except the bottleneck is concatenated with the
        decoder tensor at the matching zoom along the final feature axis.

        Tensors use the framework layout ``(b, v, t, n, d, f)``.
        """
        super().__init__(mgrids)

        del kwargs
        self.input_zoom = int(input_zoom)
        self.bottleneck_zoom = int(bottleneck_zoom)
        self.levels = list(
            range(self.input_zoom, self.bottleneck_zoom - 1, -1)
        )
        self.features = [int(value) for value in features]
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.in_zooms = [self.input_zoom]
        self.n_groups_variables = [int(value) for value in n_groups_variables]
        self.num_groups = self._expand_per_level(num_groups)
        self.num_groups_by_zoom = dict(zip(self.levels, self.num_groups))
        self.blocks_per_level = self._expand_per_level(blocks_per_level)
        self.blocks_per_level_by_zoom = dict(
            zip(self.levels, self.blocks_per_level)
        )
        self.add_refined_input: Optional[BlockWrapOperation] = (
            create_block_wrap_operation(
                add_refined_input,
                grid_layers=self.grid_layers,
            )
            if add_refined_input is not None
            else None
        )

        self._validate_configuration()

        block_kwargs = {
            "share_weights": False,
            "n_groups_variables": self.n_groups_variables,
            "rank_space": rank_space,
            "rank_time": rank_time,
            "rank_depth": rank_depth,
            "fac_mode": fac_mode,
            "grid_layers": self.grid_layers,
            "use_neighborhood": use_neighborhood,
            "norm": norm,
            "act": act,
            "residual": residual,
            "eps": eps,
            "layer_confs": {} if layer_confs is None else layer_confs,
        }

        self.stem = self._make_block(
            self.input_zoom,
            self.input_zoom,
            self.in_features,
            self.features[0],
            self.num_groups[0],
            block_kwargs,
        )

        self.encoder_refinement_blocks = nn.ModuleList(
            nn.ModuleList(
                self._make_block(
                    zoom,
                    zoom,
                    level_features,
                    level_features,
                    self.num_groups[level_index],
                    block_kwargs,
                )
                for _ in range(self.blocks_per_level[level_index] - 1)
            )
            for level_index, (zoom, level_features) in enumerate(
                zip(self.levels, self.features)
            )
        )

        self.encoder_blocks = nn.ModuleList(
            self._make_block(
                in_zoom,
                target_zoom,
                self.features[index],
                self.features[index + 1],
                self.num_groups[index + 1],
                block_kwargs,
            )
            for index, (in_zoom, target_zoom) in enumerate(
                zip(self.levels[:-1], self.levels[1:])
            )
        )

        self.decoder_up_blocks = nn.ModuleList()
        self.decoder_fusion_blocks = nn.ModuleList()
        self.decoder_refinement_blocks = nn.ModuleList()
        for target_index in range(len(self.levels) - 2, -1, -1):
            in_zoom = self.levels[target_index + 1]
            target_zoom = self.levels[target_index]
            target_features = self.features[target_index]

            self.decoder_up_blocks.append(
                self._make_block(
                    in_zoom,
                    target_zoom,
                    self.features[target_index + 1],
                    target_features,
                    self.num_groups[target_index],
                    block_kwargs,
                )
            )
            self.decoder_fusion_blocks.append(
                self._make_block(
                    target_zoom,
                    target_zoom,
                    2 * target_features,
                    target_features,
                    self.num_groups[target_index],
                    block_kwargs,
                )
            )
            self.decoder_refinement_blocks.append(
                nn.ModuleList(
                    self._make_block(
                        target_zoom,
                        target_zoom,
                        target_features,
                        target_features,
                        self.num_groups[target_index],
                        block_kwargs,
                    )
                    for _ in range(
                        self.blocks_per_level[target_index] - 1
                    )
                )
            )

        output_block_kwargs = dict(block_kwargs)
        output_block_kwargs["residual"] = False
        self.output_head = self._make_block(
            self.input_zoom,
            self.input_zoom,
            self.features[0],
            self.out_features,
            self.num_groups[0],
            output_block_kwargs,
        )

    def _expand_per_level(
        self,
        value: Union[int, Sequence[int]],
    ) -> List[int]:
        if isinstance(value, SequenceCollection) and not isinstance(
            value, (str, bytes)
        ):
            return [int(item) for item in value]
        return [int(value)] * len(self.levels)

    def _validate_configuration(self) -> None:
        if self.input_zoom <= self.bottleneck_zoom:
            raise ValueError(
                "`input_zoom` must be greater than `bottleneck_zoom`, got "
                f"{self.input_zoom} and {self.bottleneck_zoom}."
            )
        if self.bottleneck_zoom < 0:
            raise ValueError(
                f"`bottleneck_zoom` must be non-negative, got {self.bottleneck_zoom}."
            )
        if self.input_zoom > self.zoom_max:
            raise ValueError(
                f"`input_zoom` ({self.input_zoom}) exceeds the available grid maximum "
                f"({self.zoom_max})."
            )
        if len(self.features) != len(self.levels):
            raise ValueError(
                "`features` must contain one value per zoom from input to bottleneck: "
                f"expected {len(self.levels)}, got {len(self.features)}."
            )
        if any(value <= 0 for value in self.features):
            raise ValueError(
                f"`features` must contain only positive values, got {self.features}."
            )
        if len(self.num_groups) != len(self.levels):
            raise ValueError(
                "`num_groups` must be a scalar or contain one value per zoom from "
                f"input to bottleneck: expected {len(self.levels)}, got "
                f"{len(self.num_groups)}."
            )
        if any(value <= 0 for value in self.num_groups):
            raise ValueError(
                "`num_groups` must contain only positive values, got "
                f"{self.num_groups}."
            )
        if len(self.blocks_per_level) != len(self.levels):
            raise ValueError(
                "`blocks_per_level` must be a scalar or contain one value per zoom "
                f"from input to bottleneck: expected {len(self.levels)}, got "
                f"{len(self.blocks_per_level)}."
            )
        if any(value <= 0 for value in self.blocks_per_level):
            raise ValueError(
                "`blocks_per_level` must contain only positive values, got "
                f"{self.blocks_per_level}."
            )
        if self.in_features <= 0 or self.out_features <= 0:
            raise ValueError(
                "`in_features` and `out_features` must be positive, got "
                f"{self.in_features} and {self.out_features}."
            )
        if (
            self.add_refined_input is not None
            and self.in_features != self.out_features
        ):
            raise ValueError(
                "`add_refined_input` requires matching input and output features, "
                f"got {self.in_features} and {self.out_features}."
            )
        if not self.n_groups_variables or any(
            value <= 0 for value in self.n_groups_variables
        ):
            raise ValueError(
                "`n_groups_variables` must contain positive values, got "
                f"{self.n_groups_variables}."
            )
        missing_zooms = [
            zoom for zoom in self.levels if str(zoom) not in self.grid_layers
        ]
        if missing_zooms:
            raise ValueError(
                f"Missing grid layers for configured zooms {missing_zooms}."
            )

    @staticmethod
    def _make_block(
        in_zoom: int,
        target_zoom: int,
        in_features: int,
        target_features: int,
        num_groups: int,
        block_kwargs: Mapping[str, Any],
    ) -> MultiZoomHealpixConvBase:
        return MultiZoomHealpixConvBase(
            x_zooms=[in_zoom],
            in_zooms=[in_zoom],
            target_zooms=[target_zoom],
            out_zooms=[target_zoom],
            in_features=[in_features],
            target_features=[target_features],
            num_groups=num_groups,
            **block_kwargs,
        )

    def _normalize_inputs(
        self,
        x_zooms_groups: Optional[Sequence[Mapping[int, torch.Tensor]]],
    ) -> List[Dict[int, torch.Tensor]]:
        if x_zooms_groups is None or len(x_zooms_groups) == 0:
            raise ValueError("HealpixUNet requires at least one input group.")
        if len(self.n_groups_variables) not in (1, len(x_zooms_groups)):
            raise ValueError(
                f"Expected {len(self.n_groups_variables)} input groups, got "
                f"{len(x_zooms_groups)}."
            )

        normalized_groups: List[Dict[int, torch.Tensor]] = []
        for group_index, group in enumerate(x_zooms_groups):
            normalized = {int(zoom): tensor for zoom, tensor in group.items()}
            if self.input_zoom not in normalized:
                raise ValueError(
                    f"Input group {group_index} is missing zoom {self.input_zoom}."
                )
            normalized_groups.append({self.input_zoom: normalized[self.input_zoom]})
        return normalized_groups

    @staticmethod
    def _save_zoom(
        x_zooms_groups: Sequence[Mapping[int, torch.Tensor]],
        zoom: int,
    ) -> List[torch.Tensor]:
        return [group[zoom] for group in x_zooms_groups]

    @staticmethod
    def _concatenate_skip(
        x_zooms_groups: Sequence[Mapping[int, torch.Tensor]],
        skipped: Sequence[torch.Tensor],
        zoom: int,
    ) -> List[Dict[int, torch.Tensor]]:
        if len(x_zooms_groups) != len(skipped):
            raise ValueError(
                "Decoder and skip connections contain different group counts."
            )

        concatenated: List[Dict[int, torch.Tensor]] = []
        for group_index, (group, skip) in enumerate(
            zip(x_zooms_groups, skipped)
        ):
            current = group[zoom]
            if current.shape[:-1] != skip.shape[:-1]:
                raise ValueError(
                    f"Cannot concatenate skip for group {group_index}, zoom {zoom}: "
                    f"shapes {tuple(current.shape)} and {tuple(skip.shape)} differ "
                    "outside the feature axis."
                )
            concatenated.append(
                {zoom: torch.cat((current, skip), dim=-1)}
            )
        return concatenated

    def forward(
        self,
        x_zooms_groups: Optional[Sequence[Mapping[int, torch.Tensor]]] = None,
        mask_zooms_groups: Optional[
            Sequence[Optional[Mapping[int, torch.Tensor]]]
        ] = None,
        emb_groups: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
        sample_configs: Optional[Mapping[int, Any]] = None,
        out_zoom: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Dict[int, torch.Tensor]]:
        """Run the encoder, concatenating decoder skips at matching zooms."""
        del kwargs

        if out_zoom is not None and int(out_zoom) != self.input_zoom:
            raise ValueError(
                f"HealpixUNet only outputs zoom {self.input_zoom}, got "
                f"out_zoom={out_zoom}."
            )

        sample_configs = {} if sample_configs is None else sample_configs
        x_zooms_groups = self._normalize_inputs(x_zooms_groups)
        residual_state = None
        residual_context = BlockWrapContext(
            mask_groups=mask_zooms_groups,
            emb_groups=emb_groups,
            sample_configs=sample_configs,
        )
        if self.add_refined_input is not None:
            x_zooms_groups, residual_state = self.add_refined_input.pre(
                x_zooms_groups,
                residual_context,
            )
        x_zooms_groups = self.stem(
            x_zooms_groups,
            sample_configs=sample_configs,
            emb_groups=emb_groups,
        )
        for block in self.encoder_refinement_blocks[0]:
            x_zooms_groups = block(
                x_zooms_groups,
                sample_configs=sample_configs,
                emb_groups=emb_groups,
            )

        skips = {
            self.input_zoom: self._save_zoom(
                x_zooms_groups, self.input_zoom
            )
        }
        for level_index, (block, target_zoom) in enumerate(
            zip(self.encoder_blocks, self.levels[1:]),
            start=1,
        ):
            x_zooms_groups = block(
                x_zooms_groups,
                sample_configs=sample_configs,
                emb_groups=emb_groups,
            )
            for refinement_block in self.encoder_refinement_blocks[level_index]:
                x_zooms_groups = refinement_block(
                    x_zooms_groups,
                    sample_configs=sample_configs,
                    emb_groups=emb_groups,
                )
            if target_zoom != self.bottleneck_zoom:
                skips[target_zoom] = self._save_zoom(
                    x_zooms_groups, target_zoom
                )

        decoder_zooms = reversed(self.levels[:-1])
        for up_block, fusion_block, refinement_blocks, target_zoom in zip(
            self.decoder_up_blocks,
            self.decoder_fusion_blocks,
            self.decoder_refinement_blocks,
            decoder_zooms,
        ):
            x_zooms_groups = up_block(
                x_zooms_groups,
                sample_configs=sample_configs,
                emb_groups=emb_groups,
            )
            x_zooms_groups = self._concatenate_skip(
                x_zooms_groups,
                skips[target_zoom],
                target_zoom,
            )
            x_zooms_groups = fusion_block(
                x_zooms_groups,
                sample_configs=sample_configs,
                emb_groups=emb_groups,
            )
            for refinement_block in refinement_blocks:
                x_zooms_groups = refinement_block(
                    x_zooms_groups,
                    sample_configs=sample_configs,
                    emb_groups=emb_groups,
                )

        x_zooms_groups = self.output_head(
            x_zooms_groups,
            sample_configs=sample_configs,
            emb_groups=emb_groups,
        )
        if self.add_refined_input is not None:
            x_zooms_groups = self.add_refined_input.post(
                x_zooms_groups,
                residual_state,
                residual_context,
            )
        return x_zooms_groups
