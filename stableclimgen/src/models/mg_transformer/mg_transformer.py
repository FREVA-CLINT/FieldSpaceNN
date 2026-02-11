from typing import Any, Dict, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from ...utils.helpers import check_get
from ...modules.field_space.field_space_base import DiffDecoder
from ...modules.grids.grid_utils import decode_zooms
from .mg_base_model import MG_base_model, create_encoder_decoder_block, defaults

class MG_Transformer(MG_base_model):
    """
    Multi-grid transformer composed of configurable encoder/decoder blocks.
    """

    def __init__(
        self,
        mgrids: Sequence[Mapping[str, Any]],
        block_configs: Mapping[str, Any],
        in_zooms: Sequence[int],
        in_features: int = 1,
        n_groups_variables: Sequence[int] = [1],
        **kwargs: Any,
    ) -> None:
        """
        Initialize the multi-grid transformer and its block stack.

        :param mgrids: Multi-grid configuration used by the base model.
        :param block_configs: Mapping of block configurations.
        :param in_zooms: Input zoom levels used by the model.
        :param in_features: Number of input features per variable.
        :param n_groups_variables: Number of variable groups for attention layers.
        :param kwargs: Additional arguments forwarded to block factories.
        :return: None.
        """
        super().__init__(mgrids)

        self.in_zooms: Sequence[int] = in_zooms
        self.in_features: int = in_features 
       

        self.Blocks: nn.ModuleDict = nn.ModuleDict()

        in_features = [in_features]*len(in_zooms)


        for block_key, block_conf in block_configs.items():
            assert isinstance(block_key, str), "block keys should be strings"
            block = create_encoder_decoder_block(block_conf, in_zooms, in_features, n_groups_variables, self.grid_layers)

            self.Blocks[block_key] = block     

            in_features = block.out_features
            in_zooms = block.out_zooms        

        block.out_features = [in_features[0]]
        
        self.masked_residual: bool = check_get([kwargs, defaults], "masked_residual")
        self.learn_residual: bool = check_get([kwargs, defaults], "learn_residual") if not self.masked_residual else True

        self.decoder: DiffDecoder = DiffDecoder()

        
    def decode(
        self,
        x_zooms: Dict[int, torch.Tensor],
        sample_configs: Mapping[int, Any],
        out_zoom: Optional[int] = None,
        emb: Optional[Mapping[str, Any]] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Decode a single zoom-mapping into a requested zoom.

        :param x_zooms: Mapping from zoom level to tensor of shape ``(b, v, t, n, d, f)``.
        :param sample_configs: Sampling configuration dictionary per zoom.
        :param out_zoom: Optional target zoom level to decode outputs into.
        :param emb: Optional embedding dictionary for decoding.
        :return: Decoded zoom mapping.
        """
        emb = emb or {}
        return self.decoder(x_zooms, emb=emb, sample_configs=sample_configs, out_zoom=out_zoom)


    def forward(
        self,
        x_zooms_groups: Optional[Sequence[Dict[int, torch.Tensor]]] = None,
        mask_zooms_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        emb_groups: Optional[Sequence[Dict[str, Any]]] = None,
        sample_configs: Mapping[int, Any] = {},
        out_zoom: Optional[int] = None,
    ) -> Sequence[Dict[int, torch.Tensor]]:
        """
        Forward pass through the multi-grid transformer.

        :param x_zooms_groups: List of per-group zoom mappings with tensors of shape
            ``(b, v, t, n, d, f)``.
        :param mask_zooms_groups: Optional list of mask mappings aligned with inputs, each
            matching the data tensor shape ``(b, v, t, n, d, f)``.
        :param emb_groups: Optional list of embedding dictionaries aligned with inputs.
        :param sample_configs: Sampling configuration dictionary per zoom.
        :param out_zoom: Optional target zoom level to decode outputs into.
        :return: Output zoom-group mappings.
        """

        if x_zooms_groups is None:
            x_zooms_groups = []
        if mask_zooms_groups is None:
            mask_zooms_groups = [None] * len(x_zooms_groups)
        if emb_groups is None:
            emb_groups = [{} for _ in range(len(x_zooms_groups))]

        x_zooms_groups_res = None
        # Keep a residual copy for optional additive updates.
        x_zooms_groups_res = [x.copy() for x in x_zooms_groups]

        mask_zooms_groups_res = None
        if self.masked_residual:
            mask_zooms_groups_res = [
                mask.copy() if mask is not None else None for mask in mask_zooms_groups
            ]

        for block in self.Blocks.values():
            x_zooms_groups = block(
                x_zooms_groups,
                sample_configs=sample_configs,
                mask_groups=mask_zooms_groups,
                emb_groups=emb_groups,
            )

        for i, x_zooms in enumerate(x_zooms_groups):

            x_res = x_zooms_groups_res[i] if self.learn_residual else None
            x_zooms_groups[i] = self.apply_residuals(x_zooms, x_res, mask_zooms_groups, mask_zooms_groups_res, sample_configs)

        if out_zoom is not None:
            for i, x_zooms in enumerate(x_zooms_groups):
                x_zooms_groups[i] = (
                    self.decoder(x_zooms, sample_configs=sample_configs, out_zoom=out_zoom)
                    if x_zooms
                    else {}
                )

        return x_zooms_groups


    def apply_residuals(
        self,
        x_zooms: Dict[int, torch.Tensor],
        x_zooms_res: Optional[Dict[int, torch.Tensor]],
        mask_zooms: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]],
        mask_res: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]],
        sample_configs: Mapping[int, Any],
    ) -> Dict[int, torch.Tensor]:
        """
        Apply residual connections to the decoded zoom tensors.

        :param x_zooms: Current zoom mapping to update with tensors of shape
            ``(b, v, t, n, d, f)``.
        :param x_zooms_res: Residual zoom mapping (optional) with tensors of shape
            ``(b, v, t, n, d, f)``.
        :param mask_zooms: Mask mappings aligned with current outputs.
        :param mask_res: Mask mappings aligned with residuals.
        :param sample_configs: Sampling configuration dictionary per zoom.
        :return: Updated zoom mapping.
        """
        if not x_zooms:
            return x_zooms

        if mask_res is None:
            mask_res = mask_zooms

        if self.learn_residual:
            for i, x_zooms_res in enumerate(x_zooms_res):
                if len(x_zooms_res) == 1 :
                    x_zooms_res = decode_zooms(
                        x_zooms_res,
                        sample_configs=sample_configs,
                        out_zoom=list(x_zooms_res.keys())[0],
                    )

        elif self.learn_residual:
            for zoom in x_zooms.keys():
                if not self.masked_residual or mask_zooms is None:
                    x_zooms[zoom] = x_zooms_res[zoom] + x_zooms[zoom]
                elif mask_zooms[zoom].dtype == torch.bool:
                    x_zooms[zoom] = (1 - 1. * mask_zooms[zoom]) * x_zooms_res[zoom] + (mask_zooms[zoom]) * x_zooms[zoom]
                else:
                    x_zooms[zoom] = mask_zooms[zoom] * x_zooms_res[zoom] + (1 - mask_res[zoom]) * x_zooms[zoom]

        return x_zooms
