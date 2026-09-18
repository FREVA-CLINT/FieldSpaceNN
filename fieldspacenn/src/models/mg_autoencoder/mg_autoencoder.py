from typing import Any, Dict, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from ..mg_transformer.mg_base_model import MG_base_model, create_encoder_decoder_block, create_missing_zooms
from ..mg_transformer.mg_transformer import BlockExecutionStage
from ..mg_transformer.block_wrap_operations import (
    BlockWrapConfig,
    BlockWrapContext,
    create_block_wrap_operation,
)
from ...modules.field_space.field_space_base import DiffDecoder

class MG_AutoEncoder(MG_base_model):
    """
    Multi-grid autoencoder composed of configurable encoder and decoder blocks.
    """

    def __init__(
        self,
        mgrids: Any,
        in_zooms: Sequence[int],
        encoder_block_configs: Mapping[str, Any],
        decoder_block_configs: Mapping[str, Any],
        in_features: int = 1,
        out_features: int = 1,
        n_groups_variables: Sequence[int] = [1],
        **kwargs: Any,
    ) -> None:
        """
        Initialize a multi-grid autoencoder with configurable encoder/decoder blocks.

        :param mgrids: Multi-grid configuration used by the base model.
        :param in_zooms: Input zoom levels used by the model.
        :param encoder_block_configs: Mapping of encoder block configurations.
        :param decoder_block_configs: Mapping of decoder block configurations.
        :param in_features: Number of input features per variable.
        :param out_features: Number of output features per variable.
        :param n_groups_variables: Number of variable groups for each input group.
        :param kwargs: Additional arguments forwarded to block factories.
        :return: None.
        """
        super().__init__(mgrids)
        self.max_zoom: int = max(in_zooms)
        self.in_zooms: Sequence[int] = in_zooms

        self.in_features: int = in_features 


        self.out_features: int = out_features
        self.n_groups_variables = list(n_groups_variables)

        in_features = [in_features] * len(in_zooms)
        self.encoder_blocks, in_zooms, in_features = self._build_block_stack(
            encoder_block_configs,
            in_zooms,
            in_features,
            self.n_groups_variables,
            kwargs,
        )

        self.bottleneck_zooms: Sequence[int] = in_zooms

        self.decoder_blocks, in_zooms, in_features = self._build_block_stack(
            decoder_block_configs,
            in_zooms,
            in_features,
            self.n_groups_variables,
            kwargs,
        )

        self.decoder: DiffDecoder = DiffDecoder()

    def _build_block_stack(
        self,
        block_configs: Mapping[str, Any],
        in_zooms: Sequence[int],
        in_features: Sequence[int],
        n_groups_variables: Sequence[int],
        block_build_kwargs: Mapping[str, Any],
    ) -> tuple[nn.ModuleDict, Sequence[int], Sequence[int]]:
        modules = nn.ModuleDict()
        current_in_zooms = list(in_zooms)
        current_in_features = list(in_features)
        current_n_groups_variables = list(n_groups_variables)
        n_groups_depths = list(
            block_build_kwargs.get(
                "n_groups_depths",
                [1] * len(n_groups_variables),
            )
        )

        for block_key, block_conf in block_configs.items():
            assert isinstance(block_key, str), "block keys should be strings"

            if isinstance(block_conf, BlockWrapConfig):
                stage_block_configs = getattr(block_conf, "block_configs", None)
                if not stage_block_configs:
                    raise ValueError(
                        "Autoencoder BlockWrapConfig entries must define nested block_configs."
                    )

                stage_build_kwargs = dict(block_build_kwargs)
                stage_build_kwargs.update(
                    block_conf.get_block_build_overrides(
                        n_groups_variables=current_n_groups_variables,
                        n_groups_depths=n_groups_depths,
                        base_block_kwargs=stage_build_kwargs,
                    )
                )
                stage_in_zooms = block_conf.get_stage_input_zooms(current_in_zooms)
                stage_in_features = block_conf.get_stage_input_features(
                    current_in_zooms=current_in_zooms,
                    current_in_features=current_in_features,
                )
                stage_blocks = nn.ModuleDict()
                current_in_zooms = list(stage_in_zooms)
                current_in_features = list(stage_in_features)
                stage_n_groups_variables = list(
                    stage_build_kwargs.pop(
                        "n_groups_variables", current_n_groups_variables
                    )
                )
                stage_n_groups_depths = list(
                    stage_build_kwargs.pop("n_groups_depths", n_groups_depths)
                )
                stage_initialize_indexed_variables_with_same_values = list(
                    stage_build_kwargs.pop(
                        "initialize_indexed_variables_with_same_values",
                        [True] * len(stage_n_groups_variables),
                    )
                )
                stage_initialize_indexed_depths_with_same_values = list(
                    stage_build_kwargs.pop(
                        "initialize_indexed_depths_with_same_values",
                        [True] * len(stage_n_groups_variables),
                    )
                )
                stage_initialize_indexed_space_with_same_values = list(
                    stage_build_kwargs.pop(
                        "initialize_indexed_space_with_same_values",
                        [True] * len(stage_n_groups_variables),
                    )
                )
                for stage_block_key, stage_block_conf in stage_block_configs.items():
                    stage_block = create_encoder_decoder_block(
                        stage_block_conf,
                        current_in_zooms,
                        current_in_features,
                        stage_n_groups_variables,
                        self.grid_layers,
                        stage_n_groups_depths,
                        stage_initialize_indexed_variables_with_same_values,
                        stage_initialize_indexed_depths_with_same_values,
                        stage_initialize_indexed_space_with_same_values,
                        **stage_build_kwargs,
                    )
                    stage_blocks[stage_block_key] = stage_block
                    current_in_zooms = list(stage_block.out_zooms)
                    current_in_features = list(stage_block.out_features)
                    stage_n_groups_variables = list(
                        getattr(
                            stage_block,
                            "n_groups_variables_out",
                            getattr(
                                stage_block,
                                "n_groups_variables",
                                stage_n_groups_variables,
                            ),
                        )
                    )

                modules[block_key] = BlockExecutionStage(
                    wrap_operations={
                        block_key: create_block_wrap_operation(
                            block_conf,
                            grid_layers=self.grid_layers,
                        )
                    },
                    blocks=stage_blocks,
                )
                current_n_groups_variables = stage_n_groups_variables
                continue

            block = create_encoder_decoder_block(
                block_conf,
                current_in_zooms,
                current_in_features,
                current_n_groups_variables,
                grid_layers=self.grid_layers,
                **block_build_kwargs,
            )
            modules[block_key] = block
            current_in_zooms = list(block.out_zooms)
            current_in_features = list(block.out_features)
            current_n_groups_variables = list(
                getattr(
                    block,
                    "n_groups_variables_out",
                    getattr(
                        block,
                        "n_groups_variables",
                        current_n_groups_variables,
                    ),
                )
            )

        self.n_groups_variables = current_n_groups_variables
        return modules, current_in_zooms, current_in_features

    @staticmethod
    def _run_block_stack(
        blocks: nn.ModuleDict,
        x_zooms_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        sample_configs: Mapping[int, Any],
        mask_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]],
        emb_groups: Optional[Sequence[Dict[str, Any]]],
    ) -> Sequence[Optional[Dict[int, torch.Tensor]]]:
        context = BlockWrapContext(
            mask_groups=mask_groups,
            emb_groups=emb_groups,
            sample_configs=sample_configs,
        )
        for block in blocks.values():
            if isinstance(block, BlockExecutionStage):
                x_zooms_groups = block(x_zooms_groups, context)
            else:
                x_zooms_groups = block(
                    x_zooms_groups,
                    sample_configs=context.sample_configs,
                    mask_groups=context.mask_groups,
                    emb_groups=context.emb_groups,
                )
        return x_zooms_groups

    def ae_encode(
        self,
        x_zooms_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        sample_configs: Mapping[int, Any] = {},
        mask_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        emb_groups: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Sequence[Optional[Dict[int, torch.Tensor]]]:
        """
        Run the encoder stack over multi-grid input groups.

        :param x_zooms_groups: List of per-group zoom mappings with tensors of shape
            ``(b, v, t, n, d, f)``.
        :param sample_configs: Sampling configuration dictionary per zoom.
        :param mask_groups: Optional list of mask mappings aligned with ``x_zooms_groups``.
        :param emb_groups: Optional list of embedding dictionaries aligned with inputs.
        :return: Encoded zoom-group mappings.
        """
        return self._run_block_stack(
            self.encoder_blocks,
            x_zooms_groups,
            sample_configs,
            mask_groups,
            emb_groups,
        )

    def ae_decode(
        self,
        x_zooms_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        sample_configs: Mapping[int, Any] = {},
        mask_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        emb_groups: Optional[Sequence[Dict[str, Any]]] = None,
        out_zoom: Optional[int] = None,
    ) -> Sequence[Optional[Dict[int, torch.Tensor]]]:
        """
        Run the decoder stack over encoded zoom-group inputs.

        :param x_zooms_groups: List of per-group zoom mappings with tensors of shape
            ``(b, v, t, n, d, f)``.
        :param sample_configs: Sampling configuration dictionary per zoom.
        :param mask_groups: Optional list of mask mappings aligned with ``x_zooms_groups``.
        :param emb_groups: Optional list of embedding dictionaries aligned with inputs.
        :param out_zoom: Optional target zoom level to decode outputs into.
        :return: Decoded zoom-group mappings.
        """
        x_zooms_groups = self._run_block_stack(
            self.decoder_blocks,
            x_zooms_groups,
            sample_configs,
            mask_groups,
            emb_groups,
        )
        
        if out_zoom is not None:
            # Optionally decode to a single requested zoom after the decoder stack.
            for i, x_zooms in enumerate(x_zooms_groups):
                x_zooms_groups[i] = (
                    self.decoder(x_zooms, sample_configs=sample_configs, out_zoom=out_zoom)
                    if x_zooms
                    else {}
                )
        return x_zooms_groups

    def forward(
        self,
        x_zooms_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        sample_configs: Mapping[int, Any] = {},
        mask_zooms_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        emb_groups: Optional[Sequence[Dict[str, Any]]] = None,
        out_zoom: Optional[int] = None,
    ) -> Sequence[Optional[Dict[int, torch.Tensor]]]:

        """
        Forward pass for the multi-grid autoencoder.

        :param x_zooms_groups: List of per-group zoom mappings with tensors of shape
            ``(b, v, t, n, d, f)``.
        :param sample_configs: Sampling configuration dictionary per zoom.
        :param mask_zooms_groups: Optional list of mask mappings aligned with inputs.
        :param emb_groups: Optional list of embedding dictionaries aligned with inputs.
        :param out_zoom: Optional target zoom level to decode outputs into.
        :return: Decoded zoom-group mappings aligned with ``out_zoom`` when provided.
        """
        x_zooms_groups, mask_zooms_groups, emb_groups, sample_configs = create_missing_zooms(
            x_zooms_groups, self.in_zooms, mask_zooms_groups, emb_groups, sample_configs=sample_configs)

        posterior_zooms_groups = self.ae_encode(x_zooms_groups, sample_configs=sample_configs, mask_groups=mask_zooms_groups, emb_groups=emb_groups)

        dec = self.ae_decode(posterior_zooms_groups, sample_configs=sample_configs, mask_groups=mask_zooms_groups, emb_groups=emb_groups, out_zoom=out_zoom)

        return dec
