from typing import Any, Dict, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from ..mg_transformer.mg_base_model import MG_base_model, create_encoder_decoder_block
from ..mg_transformer.mg_transformer import DiffDecoder
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

        # Construct blocks based on configurations
        self.encoder_blocks: nn.ModuleDict = nn.ModuleDict()
        self.decoder_blocks: nn.ModuleDict = nn.ModuleDict()

        in_features = [in_features]*len(in_zooms)

        # Build encoder blocks, tracking output feature/zoom changes.
        for block_key, block_conf in encoder_block_configs.items():
            assert isinstance(block_key, str), "block keys should be strings"
            block = create_encoder_decoder_block(block_conf, in_zooms, in_features, n_groups_variables, self.grid_layers, **kwargs)

            self.encoder_blocks[block_key] = block

            in_features = block.out_features
            in_zooms = block.out_zooms

        self.bottleneck_zooms: Sequence[int] = in_zooms

        # Build decoder blocks, tracking output feature/zoom changes.
        for block_key, block_conf in decoder_block_configs.items():
            assert isinstance(block_key, str), "block keys should be strings"
            block = create_encoder_decoder_block(block_conf, in_zooms, in_features, n_groups_variables, self.grid_layers, **kwargs)
            self.decoder_blocks[block_key] = block

            in_features = block.out_features
            in_zooms = block.out_zooms

        block.out_features = [in_features[0]]
        
        self.decoder: DiffDecoder = DiffDecoder()

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
        for k, block in enumerate(self.encoder_blocks.values()):
            x_zooms_groups = block(x_zooms_groups, sample_configs=sample_configs, mask_groups=mask_groups, emb_groups=emb_groups)
        return x_zooms_groups

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
        for k, block in enumerate(self.decoder_blocks.values()):
            x_zooms_groups = block(x_zooms_groups, sample_configs=sample_configs, mask_groups=mask_groups, emb_groups=emb_groups)
        
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

        posterior_zooms_groups = self.ae_encode(x_zooms_groups, sample_configs=sample_configs, mask_groups=mask_zooms_groups, emb_groups=emb_groups)

        dec = self.ae_decode(posterior_zooms_groups, sample_configs=sample_configs, mask_groups=mask_zooms_groups, emb_groups=emb_groups, out_zoom=out_zoom)

        return dec
