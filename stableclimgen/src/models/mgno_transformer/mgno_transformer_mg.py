import torch
import torch.nn as nn

from omegaconf import ListConfig
from typing import List

from ...utils.helpers import check_get
from ...modules.base import LinEmbLayer
from ...modules.neural_operator import mgno_encoder_decoder as enc_dec
from ...modules.multi_grid.processing import MG_SingleBlock
from ...modules.embedding.embedder import get_embedder

from .confs import defaults, MGNOEncoderDecoderConfig, MGNOStackedEncoderDecoderConfig
from ...modules.multi_grid.confs import MGProcessingConfig

from .mgno_base_model import MGNO_base_model

class MGNO_Transformer_MG(MGNO_base_model):
    def __init__(self,
                 mgrids,
                 block_configs: List,
                 in_features: int=1,
                 lift_features: int=1,
                 out_features: int=1,
                 **kwargs
                 ) -> None: 
        
        self.max_zoom = kwargs.get("max_zoom", mgrids[-1]['zoom'])
        self.in_features = in_features 
        
        predict_var = kwargs.get("predict_var", defaults['predict_var'])
        
        super().__init__(mgrids,
                         rotate_coord_system=kwargs.get("rotate_coord_system", False))

        if predict_var:
            out_features = out_features * 2
            self.activation_var = nn.Softplus()

        self.out_features = out_features
        self.predict_var = predict_var

        embedder_input = get_embedder(**check_get([kwargs, defaults], "input_embed_confs"), zoom=self.max_zoom)
        self.in_layer = LinEmbLayer(
            in_features,
            lift_features,
            layer_confs = check_get([kwargs,defaults], "input_layer_confs"),
            embedder = embedder_input)
        
        self.Blocks = nn.ModuleList()

        in_zooms = [self.max_zoom]
        in_features = [lift_features]

        for block_idx, block_conf in enumerate(block_configs):
            
            if isinstance(block_conf, MGNOEncoderDecoderConfig):

                block = enc_dec.MGNO_EncoderDecoder_Block(
                                            self.rcm,
                                            in_zooms,
                                            in_features,
                                            block_conf.out_zooms,
                                            block_conf.no_zooms,
                                            block_conf.out_features,
                                            rule=block_conf.rule,
                                            no_layer_settings=check_get([block_conf,kwargs,defaults],'no_layer_settings'),
                                            block_type=check_get([block_conf,kwargs,defaults],'block_type'),
                                            with_gamma=check_get([block_conf,kwargs,defaults], "with_gamma"),
                                            embed_confs=check_get([block_conf,kwargs,defaults], "embed_confs"),
                                            omit_backtransform=check_get([block_conf,kwargs,defaults], "omit_backtransform"),
                                            layer_confs=check_get([block_conf,kwargs,defaults], "layer_confs"),
                                            concat_prev=check_get([block_conf,kwargs,defaults], "concat_prev"))  
                

            elif isinstance(block_conf, MGNOStackedEncoderDecoderConfig):
                block = enc_dec.MGNO_StackedEncoderDecoder_Block(
                                            self.rcm,
                                            in_zooms,
                                            in_features,
                                            block_conf.out_zooms,
                                            block_conf.no_zooms,
                                            block_conf.out_features,
                                            no_zoom_step=check_get([block_conf,kwargs,defaults],'no_zoom_step'),
                                            no_layer_settings=check_get([block_conf,kwargs,defaults],'no_layer_settings'),
                                            block_type=check_get([block_conf,kwargs,defaults],'block_type'),
                                            with_gamma=check_get([block_conf,kwargs,defaults], "with_gamma"),
                                            embed_confs=check_get([block_conf,kwargs,defaults], "embed_confs"),
                                            layer_confs=check_get([block_conf,kwargs,defaults], "layer_confs"),
                                            concat_prev=check_get([block_conf,kwargs,defaults], "concat_prev"))  
                
            elif isinstance(block_conf, MGProcessingConfig):
                layer_settings = block_conf.layer_settings
                layer_settings['layer_confs'] = check_get([block_conf,kwargs,defaults], "layer_confs")

                block = MG_SingleBlock(
                     self.rcm.grid_layers,
                     in_zooms,
                     layer_settings,
                     in_features,
                     block_conf.out_features)
                     
                
            self.Blocks.append(block)     

            in_features = block.out_features
            in_zooms = block.out_zooms

       # self.out_layer = nn.Linear(input_dims[0], output_dim, bias=False)
        
        embedder_output = get_embedder(**check_get([kwargs, defaults], "output_embed_confs"), zoom=self.max_zoom)

        self.out_layer = LinEmbLayer(
            in_features[0] if isinstance(in_features, list) or isinstance(in_features, ListConfig) else in_features,
            out_features,
            layer_confs = check_get([kwargs,defaults], "input_layer_confs"),
            embedder = embedder_output)

        self.learn_residual = check_get([kwargs,defaults], "learn_residual")

    def forward(self, x, coords_input, coords_output, sample_configs={}, mask=None, emb=None):

        """
        Forward pass for the ICON_Transformer model.

        :param x: Input tensor of shape (batch_size, num_cells, input_dim).
        :param coords_input: Input coordinates for x
        :param coords_output: Output coordinates for position embedding.
        :param sampled_indices_batch_dict: Dictionary of sampled indices for regional models.
        :param mask: Mask for dropping cells in the input tensor.
        :return: Output tensor of shape (batch_size, num_cells, output_dim).
        """

        b, nv, nt, n, nc = x.shape[:5]

        assert nc == self.in_features, f" the input has {nc} features, which doesnt match the numnber of specified input_features {self.in_features}"
        assert nc == self.out_features, f" the input has {nc} features, which doesnt match the numnber of specified out_features {self.out_features}"

        #x = x.view(b, nt, n, -1, self.in_features)
        #b,n,nv,nc = x.shape[:4]

        if self.learn_residual:
            x_res = x

        x = self.in_layer(x, sample_configs=sample_configs, emb=emb)

        x_zooms = {int(sample_configs['zoom'][0]): x} if 'zoom' in sample_configs.keys() else {self.max_zoom: x}
        mask_zooms = {int(sample_configs['zoom'][0]): mask} if 'zoom' in sample_configs.keys() else {self.max_zoom: mask}

        for k, block in enumerate(self.Blocks):
                        
            # Process input through the block
            x_zooms = block(x_zooms, sample_configs=sample_configs, mask_zooms=mask_zooms, emb=emb)

        x = x_zooms[int(sample_configs['zoom'][0]) if sample_configs else self.max_zoom]
        x = self.out_layer(x, emb=emb, sample_configs=sample_configs)

        x = x.view(b,nv,nt,n,-1)

        if self.learn_residual and not self.predict_var:
            x = x_res.view(x.shape) + x

        elif self.predict_var and self.learn_residual:
            x, x_var = x.chunk(2,dim=-1) 
            x = x_res.view(x.shape) + x
            x = torch.concat((x, self.activation_var(x_var)),dim=-1)

        elif self.predict_var:
            x, x_var = x.chunk(2,dim=-1) 
            x = torch.concat((x, self.activation_var(x_var)),dim=-1)

        return x