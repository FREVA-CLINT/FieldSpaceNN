import torch
import torch.nn as nn

from hydra.utils import instantiate
from omegaconf import ListConfig
from typing import List

from ...utils.helpers import check_get
from ...modules.base import LinEmbLayer,MLP_fac

from ...modules.multi_grid.mg_base import ConservativeLayer
from ...modules.multi_grid.processing import MG_SingleBlock,MG_MultiBlock
from ...modules.multi_grid.confs import MGProcessingConfig,MGSelfProcessingConfig,MGConservativeConfig
from ...modules.multi_grid.input_output import MG_Difference_Encoder, MG_Sum_Decoder, MG_Decoder, MG_Encoder

from ...modules.embedding.embedder import get_embedder

from .confs import defaults

from .mg_base_model import MG_base_model

class MG_Transformer(MG_base_model):
    def __init__(self, 
                 mgrids,
                 block_configs: List,
                 mg_encoder_config,
                 mg_decoder_config,
                 in_features: int=1,
                 out_features: int=1,
                 **kwargs
                 ) -> None: 
        
        self.max_zoom = kwargs.get("max_zoom", mgrids[-1]['zoom'])
        self.in_features = in_features 
        
        predict_var = kwargs.get("predict_var", defaults['predict_var'])
        
        super().__init__(mgrids)

        if predict_var:
            out_features = out_features * 2
            self.activation_var = nn.Softplus()

        self.out_features = out_features
        self.predict_var = predict_var

        self.Blocks = nn.ModuleList()

        in_zooms = [self.max_zoom]
        in_features = [in_features]

        if mg_encoder_config.type=='diff':

            self.encoder = MG_Difference_Encoder(
                out_zooms=mg_encoder_config.out_zooms
            )  
            
            out_features = in_features*len(self.encoder.out_zooms)
                
        elif mg_encoder_config.type=='nh_conv':
            self.encoder =  MG_Encoder(
                    self.grid_layers,
                    in_zooms[0],
                    in_features,
                    mg_encoder_config.out_features,
                    out_zooms=mg_encoder_config.out_zooms,
                    layer_confs=layer_confs)

            out_features = self.encoder.out_features

        in_zooms = self.encoder.out_zooms
        in_features = out_features

        for block_idx, block_conf in enumerate(block_configs):
            layer_confs = check_get([block_conf,kwargs,defaults], "layer_confs")
                
            if isinstance(block_conf, MGProcessingConfig):
                layer_settings = block_conf.layer_settings

                block = MG_SingleBlock(
                     self.grid_layers,
                     in_zooms,
                     layer_settings,
                     in_features,
                     block_conf.out_features,
                    layer_confs=layer_confs)
                        
            elif isinstance(block_conf, MGConservativeConfig):
                block = ConservativeLayer(in_zooms,
                                          first_feature_only=self.predict_var)
                block.out_features = in_features

            elif isinstance(block_conf, MGSelfProcessingConfig):
                layer_settings = block_conf.layer_settings
                layer_settings['layer_confs'] = check_get([block_conf,kwargs,defaults], "layer_confs")

                block = MG_MultiBlock(
                     self.grid_layers,
                     in_zooms,
                     layer_settings,
                     in_features,
                     block_conf.out_features,
                     q_zooms  = check_get([block_conf,kwargs,{"q_zooms": -1}], "q_zooms"),
                     kv_zooms = check_get([block_conf,kwargs,{"kv_zooms": -1}], "kv_zooms"),
                     layer_confs=layer_confs)
                
            self.Blocks.append(block)     

            in_features = block.out_features
            in_zooms = block.out_zooms

        if mg_decoder_config.type=='sum':

            self.decoder = MG_Sum_Decoder(
                        self.grid_layers,
                        in_zooms=in_zooms,
                        out_zoom=self.max_zoom,
                        conservative = check_get([mg_decoder_config,kwargs,{"conservative": False}], "conservative"),
                        interpolator_confs=check_get([mg_decoder_config,kwargs,defaults], "interpolator_confs")
                    )  
            
            out_features = [in_features[0]]
                
        elif mg_decoder_config.type=='nh_conv':
            self.decoder = MG_Decoder(
                                self.grid_layers,
                                in_zooms=in_zooms,
                                in_features_list=in_features,
                                out_features=mg_decoder_config.out_features,
                                out_zoom=check_get([mg_decoder_config, {"out_zoom": self.max_zoom}], "out_zoom"),
                                with_residual=check_get([mg_decoder_config,kwargs,defaults], "with_residual"),
                                layer_confs=layer_confs,
                                aggregation=check_get([mg_decoder_config,kwargs,defaults], "aggregation"),
                            ) 
        

        block.out_features = [in_features[0]]
       # self.out_layer = nn.Linear(input_dims[0], output_dim, bias=False)

        
        #self.out_layer = MLP_fac(in_features[0], out_features, layer_confs=check_get([kwargs,defaults], "input_layer_confs"))
        
        self.learn_residual = check_get([kwargs,defaults], "learn_residual")

    def forward(self, x, coords_input, coords_output, sample_dict={}, mask=None, emb=None, return_zooms=True):

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
        assert nc == (self.out_features // (1+ self.predict_var)), f" the input has {nc} features, which doesnt match the numnber of specified out_features {self.out_features}"

       # x = x.view(b, nv, nt, n, self.in_features)
        #b,n,nv,nc = x.shape[:4]

        

        #x = self.in_layer(x, sample_dict=sample_dict, emb=emb)
        #x = x.view(*x.shape[:4],-1)

        x_zooms = {int(sample_dict['zoom'][0]): x} if 'zoom' in sample_dict.keys() else {self.max_zoom: x}
        mask_zooms = {int(sample_dict['zoom'][0]): mask} if 'zoom' in sample_dict.keys() else {self.max_zoom: mask}
        
        x_zooms = self.encoder(x_zooms, emb=emb, sample_dict=sample_dict)

        if self.learn_residual:
            x_zooms_res = x_zooms

        for k, block in enumerate(self.Blocks):
                        
            # Process input through the block
            x_zooms = block(x_zooms, sample_dict=sample_dict, mask_zooms=mask_zooms, emb=emb)

        if self.learn_residual:
            for zoom in x_zooms.keys():
                x_zooms[zoom] = x_zooms_res[zoom] + x_zooms[zoom]

        elif self.predict_var and self.learn_residual:
            for zoom, x in x_zooms.items():
                x, x_var = x.chunk(2,dim=-1) 
                x = x_zooms_res[zoom] + x
                x_zooms[zoom] = torch.concat((x, self.activation_var(x_var)),dim=-1)

        elif self.predict_var:
            for zoom, x in x_zooms.items():
                x, x_var = x.chunk(2,dim=-1) 
                x_zooms[zoom] = torch.concat((x, self.activation_var(x_var)),dim=-1)


        if return_zooms:
            return x_zooms
        else:
            return self.decoder(x_zooms, emb=emb, sample_dict=sample_dict)
