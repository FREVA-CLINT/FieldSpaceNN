from typing import List,Dict

import torch
import torch.nn as nn
from stableclimgen.src.modules.distributions.distributions import DiagonalGaussianDistribution, DiracDistribution

from ..mg_transformer.confs import defaults
from ..mg_transformer.mg_base_model import MG_base_model
from .confs import MGQuantConfig
from ..mg_transformer.mg_transformer import DiffDecoder
from ...modules.embedding.embedding_layers import get_mg_embeddings
from ...modules.multi_grid.mg_base import ConservativeLayer
from ...utils.helpers import check_get
from ...modules.multi_grid.mg_base import ConservativeLayer,ConservativeLayerConfig, DiffDecoder
from ...modules.multi_grid.field_layer import FieldLayer, FieldLayerConfig
from ...modules.multi_grid.field_attention import FieldAttentionModule,FieldAttentionConfig

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
            block = self.create_encoder_decoder_block(block_conf, in_zooms, in_features, n_groups_variables, **kwargs)
                
            self.encoder_blocks[block_key] = block

            in_features = block.out_features
            in_zooms = block.out_zooms

        self.bottleneck_zooms = in_zooms

        for block_key, block_conf in decoder_block_configs.items():
            assert isinstance(block_key, str), "block keys should be strings"
            block = self.create_encoder_decoder_block(block_conf, in_zooms, in_features, n_groups_variables, **kwargs)
            self.decoder_blocks[block_key] = block

            in_features = block.out_features
            in_zooms = block.out_zooms

        block.out_features = [in_features[0]]
        
        self.decoder = DiffDecoder()


    def create_encoder_decoder_block(self, block_conf, in_zooms, in_features, n_groups_variables, **kwargs):
        embed_confs = check_get([block_conf, kwargs, defaults], "embed_confs")
        layer_confs = check_get([block_conf, kwargs, defaults], "layer_confs")
        layer_confs_emb = check_get([block_conf, kwargs, defaults], "layer_confs_emb")
        dropout = check_get([block_conf, kwargs, defaults], "dropout")
        out_zooms = check_get([block_conf, {'out_zooms':in_zooms}], "out_zooms")
        use_mask = check_get([block_conf, kwargs, defaults], "use_mask")
        n_head_channels = check_get([block_conf,kwargs,defaults], "n_head_channels")
        att_dim = check_get([block_conf,kwargs,defaults], "att_dim")

                    
        if isinstance(block_conf, ConservativeLayerConfig):
            block = ConservativeLayer(in_zooms, first_feature_only=self.predict_var)
            block.out_features = in_features
                    
        
        elif isinstance(block_conf, FieldAttentionConfig):

            block = FieldAttentionModule(
                    self.grid_layers,
                    in_zooms,
                    out_zooms,
                    token_zoom = block_conf.token_zoom,
                    q_zooms  = block_conf.q_zooms,
                    kv_zooms = block_conf.kv_zooms,
                    use_mask = use_mask,
                    refine_zooms= block_conf.refine_zooms,
                    shift= block_conf.shift,
                    multi_shift= block_conf.multi_shift,
                    att_dim = att_dim,
                    n_groups_variables = n_groups_variables,
                    token_len_time = block_conf.token_len_time,
                    token_len_depth = block_conf.token_len_depth,
                    token_overlap_space = block_conf.token_overlap_space,
                    token_overlap_time = block_conf.token_overlap_time,
                    token_overlap_depth = block_conf.token_overlap_depth,
                    token_overlap_mlp_time = block_conf.token_overlap_mlp_time,
                    token_overlap_mlp_depth = block_conf.token_overlap_mlp_depth,
                    rank_space = block_conf.rank_space,
                    rank_time = block_conf.rank_time,
                    rank_depth = block_conf.rank_depth,
                    seq_len_zoom = block_conf.seq_len_zoom,
                    seq_len_time =  block_conf.seq_len_time,
                    seq_len_depth = block_conf.seq_len_depth,
                    seq_overlap_space = block_conf.seq_overlap_space,
                    seq_overlap_time = block_conf.seq_overlap_time,
                    seq_overlap_depth = block_conf.seq_overlap_depth,
                    with_var_att= block_conf.with_var_att,
                    update = block_conf.update,
                    dropout = dropout,
                    n_head_channels = n_head_channels,
                    embed_confs = embed_confs,
                    separate_mlp_norm = block_conf.separate_mlp_norm,
                    layer_confs=layer_confs,
                    layer_confs_emb = layer_confs_emb)
            
            block.out_features = in_features


        elif isinstance(block_conf, FieldLayerConfig):
    
            block = FieldLayer(
                    self.grid_layers,
                    in_zooms,
                    block_conf.in_zooms,
                    block_conf.target_zooms,
                    block_conf.field_zoom,
                    out_zooms=block_conf.out_zooms,
                    in_features=in_features,
                    target_features=check_get([block_conf,{"target_features": in_features}], "target_features"),
                    mult = block_conf.mult,
                    overlap= block_conf.overlap,
                    type= block_conf.type,
                    layer_confs=layer_confs)
            
        return block



    def vae_encode(self, x_zooms, sample_configs={}, mask_zooms=None, emb=None):
        for k, block in enumerate(self.encoder_blocks.values()):
            x_zooms = block(x_zooms, sample_configs=sample_configs, mask_zooms=mask_zooms, emb=emb)
        return x_zooms

    def vae_decode(self, x_zooms, sample_configs={}, mask_zooms=None, emb=None, out_zoom=None):
        for k, block in enumerate(self.decoder_blocks.values()):
            x_zooms = block(x_zooms, sample_configs=sample_configs, mask_zooms=mask_zooms, emb=emb)
        x_zooms = self.decoder(x_zooms, sample_configs=sample_configs, emb=emb, out_zoom=out_zoom)
        return x_zooms

    def decode(self, x_zooms:Dict[str, torch.Tensor], sample_configs: Dict, out_zoom: int=None, emb={}):
        return self.decoder(x_zooms, emb=emb, sample_configs=sample_configs, out_zoom=out_zoom)


    def forward(self, x_zooms: Dict[int, torch.Tensor], sample_configs={}, mask_zooms: Dict[int, torch.Tensor]= None, emb=None, out_zoom=None):

        """
        Forward pass for the ICON_Transformer model.

        :param x: Input tensor of shape (batch_size, num_cells, input_dim).
        :param coords_input: Input coordinates for x
        :param coords_output: Output coordinates for position embedding.
        :param sampled_indices_batch_dict: Dictionary of sampled indices for regional models.
        :param mask: Mask for dropping cells in the input tensor.
        :return: Output tensor of shape (batch_size, num_cells, output_dim).
        """

        b, nv, nt, n, nc = x_zooms[list(x_zooms.keys())[0]].shape

        assert nc == self.in_features, f" the input has {nc} features, which doesnt match the numnber of specified input_features {self.in_features}"
        assert nc == (self.out_features // (1+ self.predict_var)), f" the input has {nc} features, which doesnt match the numnber of specified out_features {self.out_features}"

        posterior_zooms = self.vae_encode(x_zooms, sample_configs=sample_configs, mask_zooms=mask_zooms, emb=emb)

        z_zooms = {int(zoom): x.sample(gamma=self.gammas[str(zoom)] if self.gammas else None) for zoom, x in posterior_zooms.items()}

        dec = self.vae_decode(z_zooms, sample_configs=sample_configs, mask_zooms=mask_zooms, emb=emb, out_zoom=out_zoom)

        return dec, posterior_zooms