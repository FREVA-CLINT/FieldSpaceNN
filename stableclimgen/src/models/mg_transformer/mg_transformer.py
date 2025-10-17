import torch
import torch.nn as nn

from hydra.utils import instantiate
from omegaconf import ListConfig
from typing import List,Dict

from ...utils.helpers import check_get
from ...modules.base import LinEmbLayer,MLP_fac

from ...modules.embedding.embedding_layers import get_mg_embeddings
from ...modules.multi_grid.mg_base import ConservativeLayer,DecodeLayer,Conv_EncoderDecoder,MGFieldLayer
from ...modules.multi_grid.processing import MG_SingleBlock,MG_MultiBlock
from ...modules.multi_grid.confs import MGProcessingConfig,MGSelfProcessingConfig,MGFieldAttentionConfig,MGConservativeConfig,MGCoordinateEmbeddingConfig,MGDecodeConfig,FieldLayerConfig,Conv_EncoderDecoderConfig,MGChannelAttentionConfig,MGFieldLayerConfig

from ...modules.embedding.embedder import get_embedder
from ...modules.grids.grid_utils import decode_zooms

from .confs import defaults

from .mg_base_model import MG_base_model

class MG_Transformer(MG_base_model):
    def __init__(self, 
                 mgrids,
                 block_configs: List,
                 in_zooms: List,
                 in_features: int=1,
                 out_features: int=1,
                 shared_mg_emb_confs: dict={},
                 with_global_gamma=True,
                 decoder_settings= {},
                 **kwargs
                 ) -> None: 
        
        
        super().__init__(mgrids)

        self.in_zooms = in_zooms
        self.in_features = in_features 
        predict_var = kwargs.get("predict_var", defaults['predict_var'])

        if predict_var:
            out_features = out_features * 2
            self.activation_var = nn.Softplus()

        self.out_features = out_features
        self.predict_var = predict_var

        if len(shared_mg_emb_confs)>0:
            self.mg_emeddings = get_mg_embeddings(shared_mg_emb_confs, self.grid_layers)
        else:
            self.mg_emeddings = None

        self.Blocks = nn.ModuleDict()

        in_features = [in_features]*len(in_zooms)

        for block_key, block_conf in block_configs.items():
            assert isinstance(block_key, str), "block keys should be strings"

            layer_confs = check_get([block_conf,kwargs,defaults], "layer_confs")

            if isinstance(block_conf, MGProcessingConfig):
                layer_settings = block_conf.layer_settings

                block = MG_SingleBlock(
                     self.grid_layers,
                     in_zooms,
                     layer_settings,
                     in_features,
                     block_conf.out_features,
                     zooms=check_get([block_conf, kwargs, {"zooms": in_zooms}], "zooms"),
                     layer_confs=layer_confs,
                     layer_confs_emb=check_get([block_conf, kwargs, {"layer_confs_emb": {}}], "layer_confs_emb"),
                     use_mask=check_get([block_conf, kwargs,{"use_mask": False}], "use_mask"),
                     n_head_channels=check_get([block_conf,kwargs,defaults], "n_head_channels"))
                        
            elif isinstance(block_conf, MGConservativeConfig):
                block = ConservativeLayer(in_zooms,
                                          first_feature_only=self.predict_var)
                block.out_features = in_features

            elif isinstance(block_conf, MGDecodeConfig):
                block = DecodeLayer(block_conf.out_zoom)
                
                block.out_features = in_features
            

            elif isinstance(block_conf, MGSelfProcessingConfig):
                layer_settings = block_conf.layer_settings
                layer_settings['layer_confs'] = check_get([block_conf,kwargs,defaults], "layer_confs")

                block = MG_MultiBlock(
                     self.grid_layers,
                     in_zooms,
                     check_get([block_conf,{'out_zooms':in_zooms}], "out_zooms"),
                     layer_settings,
                     in_features,
                     block_conf.out_features,
                     q_zooms  = check_get([block_conf,kwargs,{"q_zooms": -1}], "q_zooms"),
                     kv_zooms = check_get([block_conf,kwargs,{"kv_zooms": -1}], "kv_zooms"),
                     layer_confs=layer_confs,
                     layer_confs_emb=check_get([block_conf,kwargs,{"layer_confs_emb": {}}], "layer_confs_emb"),
                     use_mask=check_get([block_conf, kwargs,{"use_mask": False}], "use_mask"),
                     init_missing_zooms=check_get([block_conf, kwargs,{"init_missing_zooms": "zeros"}], "init_missing_zooms"),
                     residual=check_get([block_conf, kwargs,{"residual": False}], "residual"),
                     n_head_channels=check_get([block_conf,kwargs,defaults], "n_head_channels"))
            
            elif isinstance(block_conf, MGFieldAttentionConfig):
                layer_settings = block_conf.layer_settings
                layer_settings['layer_confs'] = check_get([block_conf,kwargs,defaults], "layer_confs")

                block = MG_MultiBlock(
                     self.grid_layers,
                     in_zooms,
                     check_get([block_conf,{'out_zooms':in_zooms}], "out_zooms"),
                     layer_settings,
                     in_features,
                     block_conf.out_features,
                     q_zooms  = check_get([block_conf,kwargs,{"q_zooms": -1}], "q_zooms"),
                     kv_zooms = check_get([block_conf,kwargs,{"kv_zooms": -1}], "kv_zooms"),
                     layer_confs=layer_confs,
                     layer_confs_emb=check_get([block_conf,kwargs,{"layer_confs_emb": {}}], "layer_confs_emb"),
                     use_mask=check_get([block_conf, kwargs,{"use_mask": False}], "use_mask"),
                     type='field_att',
                     init_missing_zooms=check_get([block_conf, kwargs, {"init_missing_zooms": "zeros"}], "init_missing_zooms"),
                     residual=check_get([block_conf, kwargs,{"residual": False}], "residual"),
                     n_head_channels=check_get([block_conf,kwargs,defaults], "n_head_channels"))
            
            elif isinstance(block_conf, MGChannelAttentionConfig):
                layer_settings = block_conf.layer_settings
                layer_settings['layer_confs'] = check_get([block_conf,kwargs,defaults], "layer_confs")

                block = MG_MultiBlock(
                     self.grid_layers,
                     in_zooms,
                     check_get([block_conf,{'out_zooms':in_zooms}], "out_zooms"),
                     layer_settings,
                     in_features=1,
                     out_features=in_features,
                     q_zooms  = check_get([block_conf,kwargs,{"q_zooms": -1}], "q_zooms"),
                     kv_zooms = check_get([block_conf,kwargs,{"kv_zooms": -1}], "kv_zooms"),
                     layer_confs=layer_confs,
                     layer_confs_emb=check_get([block_conf,kwargs,{"layer_confs_emb": {}}], "layer_confs_emb"),
                     use_mask=check_get([block_conf, kwargs,{"use_mask": False}], "use_mask"),
                     type='channel_att',
                     n_head_channels=check_get([block_conf,kwargs,defaults], "n_head_channels"))

                block.out_features = in_features

            elif isinstance(block_conf, MGFieldLayerConfig):
        
                layer_confs = check_get([block_conf,kwargs,defaults], "layer_confs")

                block = MGFieldLayer(
                        self.grid_layers[str(block_conf.field_zoom)],
                        block_conf.in_zooms,
                        block_conf.target_zooms,
                        block_conf.field_zoom,
                        out_zooms=block_conf.out_zooms,
                        mult = block_conf.mult,
                        with_nh = block_conf.with_nh,
                        with_residual = block_conf.with_residual,
                        type= block_conf.type,
                        layer_confs=layer_confs)
                
                block.out_features = in_features

            elif isinstance(block_conf, Conv_EncoderDecoderConfig):
                layer_confs = check_get([block_conf, kwargs, defaults], "layer_confs")

                block = Conv_EncoderDecoder(
                    self.grid_layers,
                    in_zooms,
                    zoom_map = block_conf.zoom_map,
                    in_features_list = in_features,
                    out_zooms = check_get([block_conf, kwargs, {"out_zooms": None}], "out_zooms"),
                    aggregation=check_get([block_conf, kwargs, {"aggregation": "sum"}], "aggregation"),
                    use_skip_conv=check_get([block_conf, kwargs, {"use_skip_conv": False}], "use_skip_conv"),
                    with_gamma=check_get([block_conf, kwargs, {"with_gamma": False}], "with_gamma"),
                    layer_confs=layer_confs
                )
            self.Blocks[block_key] = block     

            in_features = block.out_features
            in_zooms = block.out_zooms        

        block.out_features = [in_features[0]]
        
        self.masked_residual = check_get([kwargs, defaults], "masked_residual")
        self.learn_residual = check_get([kwargs, defaults], "learn_residual") if not self.masked_residual else True

        self.decoder = DiffDecoder()

        
    def decode(self, x_zooms:Dict[str, torch.Tensor], sample_configs: Dict, out_zoom: int=None, emb={}):
        return self.decoder(x_zooms, emb=emb, sample_configs=sample_configs, out_zoom=out_zoom)


    def forward(self, x_zooms: Dict[int, torch.Tensor], sample_configs={}, mask_zooms: Dict[int, torch.Tensor]= None, emb=None, out_zoom=None):

        """
        Forward pass for the ICON_Transformer model.

        :param x: Input tensor of shape (batch_size, num_cells, input_dim).
        :param coords_input: Input coordinates for x
        :param coords_output: Output coordinates for position embedding.
        :param sampled_indices_batch_dict: Dictionary of sampled indices for regional models.
        :param mask_zooms: Mask for dropping cells in the input tensor.
        :return: Output tensor of shape (batch_size, num_cells, output_dim).
        """

        b, nv, nt, n, nc = x_zooms[list(x_zooms.keys())[0]].shape

        assert nc == self.in_features, f" the input has {nc} features, which doesnt match the numnber of specified input_features {self.in_features}"
        assert nc == (self.out_features // (1+ self.predict_var)), f" the input has {nc} features, which doesnt match the numnber of specified out_features {self.out_features}"

        emb['SharedMGEmbedder'] = (self.mg_emeddings, emb['GroupEmbedder'])

       # x_zooms = {int(sample_configs['zoom'][0]): x} if 'zoom' in sample_configs.keys() else {self.max_zoom: x}
       # mask_zooms = {int(sample_configs['zoom'][0]): mask} if 'zoom' in sample_configs.keys() else {self.max_zoom: mask}
        
       # x_zooms = self.encoder(x_zooms, emb=emb, sample_configs=sample_configs)
       
        if self.learn_residual:
            x_zooms_res = x_zooms.copy()
        
        if self.masked_residual:
            mask_res = mask_zooms.copy()

        for k, block in enumerate(self.Blocks.values()):
            # Process input through the block
            x_zooms = block(x_zooms, sample_configs=sample_configs, mask_zooms=mask_zooms, emb=emb)

        

        if self.learn_residual and len(x_zooms.keys())==1:
            x_zooms_res = decode_zooms(x_zooms_res, sample_configs=sample_configs, out_zoom=list(x_zooms.keys())[0])

        if self.predict_var and self.learn_residual:
            for zoom, x in x_zooms.items():
                x, x_var = x.chunk(2,dim=-1) 

                if not self.masked_residual:
                    x = x_zooms_res[zoom] + x
                elif mask_zooms[zoom].dtype == torch.bool:
                    x = (1 - mask_zooms[zoom]) * x_zooms_res[zoom] + (mask_zooms[zoom]) * x
                else:
                    x = mask_zooms[zoom] * x_zooms_res[zoom] + (1 - mask_zooms[zoom]) * x

                x_zooms[zoom] = torch.concat((x, self.activation_var(x_var)),dim=-1)

        elif self.predict_var:
            for zoom, x in x_zooms.items():
                x, x_var = x.chunk(2,dim=-1) 
                x_zooms[zoom] = torch.concat((x, self.activation_var(x_var)),dim=-1)

        elif self.learn_residual:
            for zoom in x_zooms.keys():

                if not self.masked_residual:
                    x_zooms[zoom] = x_zooms_res[zoom] + x_zooms[zoom]
                elif mask_zooms[zoom].dtype == torch.bool:
                    x_zooms[zoom] = (1 - 1.*mask_zooms[zoom]) * x_zooms_res[zoom] + (mask_zooms[zoom]) * x_zooms[zoom]
                else:
                    x_zooms[zoom] = mask_zooms[zoom] * x_zooms_res[zoom] + (1 - mask_res[zoom]) * x_zooms[zoom]

        x_zooms = self.decoder(x_zooms, sample_configs=sample_configs, emb=emb, out_zoom=out_zoom)

        return x_zooms

class DiffDecoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_zooms: dict, sample_configs: Dict, out_zoom: int = None, **kwargs):

        if out_zoom is None:
            return x_zooms
        
        return decode_zooms(x_zooms, sample_configs=sample_configs, out_zoom=out_zoom)