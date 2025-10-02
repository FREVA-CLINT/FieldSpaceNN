import torch
import torch.nn as nn

from hydra.utils import instantiate
from omegaconf import ListConfig
from typing import List,Dict

from ...utils.helpers import check_get
from ...modules.base import LinEmbLayer,MLP_fac

from ...modules.multi_grid.mg_base import ConservativeLayer,MGEmbedding,get_mg_embeddings,DecodeLayer,MFieldLayer
from ...modules.multi_grid.processing import MG_SingleBlock,MG_MultiBlock
from ...modules.multi_grid.confs import MGProcessingConfig,MGSelfProcessingConfig,MGFieldAttentionConfig,MGConservativeConfig,MGCoordinateEmbeddingConfig,MGDecodeConfig,FieldLayerConfig

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
                 mg_emb_confs: dict={},
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

        if len(mg_emb_confs)>0:
            self.mg_emeddings = get_mg_embeddings(mg_emb_confs, self.grid_layers)
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
                     check_get([block_conf, kwargs, {"out_zooms": in_zooms}], "out_zooms"),
                     layer_settings,
                     in_features,
                     block_conf.out_features,
                     layer_confs=layer_confs,
                     layer_confs_emb=check_get([block_conf, kwargs, {"layer_confs_emb": {}}], "layer_confs_emb"),
                     use_mask=check_get([block_conf, kwargs,{"use_mask": False}], "use_mask"))
                        
            elif isinstance(block_conf, MGConservativeConfig):
                block = ConservativeLayer(in_zooms,
                                          first_feature_only=self.predict_var)
                block.out_features = in_features

            elif isinstance(block_conf, MGDecodeConfig):
                block = DecodeLayer(block_conf.out_zoom)
                
                block.out_features = in_features
            
            elif isinstance(block_conf, MGCoordinateEmbeddingConfig):
                block = MGEmbedding(self.grid_layers[str(block_conf.emb_zoom)],
                                    block_conf.features,
                                    n_groups=check_get([block_conf,kwargs,{'n_groups': 1}], "n_groups"),
                                    zooms = in_zooms,
                                    init_mode=check_get([block_conf,kwargs,{'init_mode': "fourier_sphere"}], "init_mode"),
                                    layer_confs=layer_confs)
                block.out_zooms = in_zooms

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
                     residual=check_get([block_conf, kwargs,{"residual": False}], "residual"))
            
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
                     residual=check_get([block_conf, kwargs,{"residual": False}], "residual"))
                
            self.Blocks[block_key] = block     

            in_features = block.out_features
            in_zooms = block.out_zooms        

        block.out_features = [in_features[0]]
        
        self.learn_residual = check_get([kwargs,defaults], "learn_residual")

        if len(decoder_settings)==0:
            self.decoder = DiffDecoder()
        
        else:
            self.decoder = MFieldLayer(
                in_features,
                decoder_settings['out_features'],
                in_zooms,
                self.grid_layers,
                with_nh = decoder_settings.get('with_nh', True),
                embed_confs = decoder_settings.get('embed_confs', {}),
                N = decoder_settings.get('N', 2),
                kmin = decoder_settings.get('kmin', 0),
                kmax = decoder_settings.get('kmin', 0.5),
                layer_confs = decoder_settings.get('layer_confs', {}))
        
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

        emb['MGEmbedder'] = (self.mg_emeddings, emb['GroupEmbedder'])

       # x_zooms = {int(sample_configs['zoom'][0]): x} if 'zoom' in sample_configs.keys() else {self.max_zoom: x}
       # mask_zooms = {int(sample_configs['zoom'][0]): mask} if 'zoom' in sample_configs.keys() else {self.max_zoom: mask}
        
       # x_zooms = self.encoder(x_zooms, emb=emb, sample_configs=sample_configs)
       
        if self.learn_residual:
            x_zooms_res = {k: v.clone() for k, v in x_zooms.items()}

        for k, block in enumerate(self.Blocks.values()):
            # Process input through the block
            x_zooms = block(x_zooms, sample_configs=sample_configs, mask_zooms=mask_zooms, emb=emb)

        x_zooms = self.decoder(x_zooms, sample_configs=sample_configs, emb=emb, out_zoom=out_zoom)

        if self.learn_residual and len(x_zooms.keys())==1:
            x_zooms_res = decode_zooms(x_zooms_res, sample_configs=sample_configs, out_zoom=list(x_zooms.keys())[0])

        if self.predict_var and self.learn_residual:
            for zoom, x in x_zooms.items():
                x, x_var = x.chunk(2,dim=-1) 
                x = x_zooms_res[zoom] + x
                x_zooms[zoom] = torch.concat((x, self.activation_var(x_var)),dim=-1)

        elif self.predict_var:
            for zoom, x in x_zooms.items():
                x, x_var = x.chunk(2,dim=-1) 
                x_zooms[zoom] = torch.concat((x, self.activation_var(x_var)),dim=-1)

        elif self.learn_residual:
            for zoom in x_zooms.keys():
                x_zooms[zoom] = x_zooms_res[zoom] + x_zooms[zoom]

        return x_zooms

class DiffDecoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_zooms: dict, sample_configs: Dict, out_zoom: int = None, **kwargs):

        if out_zoom is None:
            return x_zooms
        
        return decode_zooms(x_zooms, sample_configs=sample_configs, out_zoom=out_zoom)