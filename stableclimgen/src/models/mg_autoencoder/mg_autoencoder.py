from typing import List,Dict

import torch
import torch.nn as nn

from ..mg_transformer.confs import defaults
from ..mg_transformer.mg_base_model import MG_base_model, create_encoder_decoder_block
from ..mg_transformer.mg_transformer import DiffDecoder
from ...modules.multi_grid.mg_base import DiffDecoder

class MG_AutoEncoder(MG_base_model):
    def __init__(self, 
                 mgrids,
                 in_zooms: List,
                 encoder_block_configs: List,
                 decoder_block_configs: List,
                 in_features: int=1,
                 out_features: int=1,
                 n_groups_variables: List = [1],
                 **kwargs
                 ) -> None: 
        
        
        super().__init__(mgrids)
        self.max_zoom = max(in_zooms)
        self.in_zooms = in_zooms

        self.in_features = in_features 
        predict_var = kwargs.get("predict_var", defaults['predict_var'])

        if predict_var:
            out_features = out_features * 2
            self.activation_var = nn.Softplus()

        self.out_features = out_features
        self.predict_var = predict_var

        # Construct blocks based on configurations
        self.encoder_blocks = nn.ModuleDict()
        self.decoder_blocks = nn.ModuleDict()

        in_features = [in_features]*len(in_zooms)

        for block_key, block_conf in encoder_block_configs.items():
            assert isinstance(block_key, str), "block keys should be strings"
            block = create_encoder_decoder_block(block_conf, in_zooms, in_features, n_groups_variables, self.predict_var, self.grid_layers, **kwargs)

            self.encoder_blocks[block_key] = block

            in_features = block.out_features
            in_zooms = block.out_zooms

        self.bottleneck_zooms = in_zooms

        for block_key, block_conf in decoder_block_configs.items():
            assert isinstance(block_key, str), "block keys should be strings"
            block = create_encoder_decoder_block(block_conf, in_zooms, in_features, n_groups_variables, self.predict_var, self.grid_layers, **kwargs)
            self.decoder_blocks[block_key] = block

            in_features = block.out_features
            in_zooms = block.out_zooms

        block.out_features = [in_features[0]]
        
        self.decoder = DiffDecoder()

    def ae_encode(self, x_zooms_groups, sample_configs={}, mask_groups=None, emb_groups=None):
        for k, block in enumerate(self.encoder_blocks.values()):
            x_zooms_groups = block(x_zooms_groups, sample_configs=sample_configs, mask_groups=mask_groups, emb_groups=emb_groups)
        return x_zooms_groups

    def ae_decode(self, x_zooms_groups, sample_configs={}, mask_groups=None, emb_groups=None, out_zoom=None):
        for k, block in enumerate(self.decoder_blocks.values()):
            x_zooms_groups = block(x_zooms_groups, sample_configs=sample_configs, mask_groups=mask_groups, emb_groups=emb_groups)
        
        if out_zoom is not None:
            for i, x_zooms in enumerate(x_zooms_groups):
                x_zooms_groups[i] = (
                    self.decoder(x_zooms, sample_configs=sample_configs, out_zoom=out_zoom)
                    if x_zooms
                    else {}
                )
        return x_zooms_groups

    def forward(self, x_zooms_groups: Dict[int, torch.Tensor], sample_configs={}, mask_zooms_groups: Dict[int, torch.Tensor]= None, emb_groups=None, out_zoom=None):

        """
        Forward pass for the ICON_Transformer model.

        :param x: Input tensor of shape (batch_size, num_cells, input_dim).
        :param coords_input: Input coordinates for x
        :param coords_output: Output coordinates for position embedding.
        :param sampled_indices_batch_dict: Dictionary of sampled indices for regional models.
        :param mask: Mask for dropping cells in the input tensor.
        :return: Output tensor of shape (batch_size, num_cells, output_dim).
        """

        posterior_zooms_groups = self.ae_encode(x_zooms_groups, sample_configs=sample_configs, mask_groups=mask_zooms_groups, emb_groups=emb_groups)

        dec = self.ae_decode(posterior_zooms_groups, sample_configs=sample_configs, mask_groups=mask_zooms_groups, emb_groups=emb_groups, out_zoom=out_zoom)

        return dec