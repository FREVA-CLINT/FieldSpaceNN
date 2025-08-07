from typing import List,Dict
import torch.nn as nn
import torch
import copy

from ...modules.grids.grid_layer import MultiRelativeCoordinateManager
from ...utils.helpers import check_get_missing_key

from .no_blocks import PreActivation_NOBlock, NOBlock, Stacked_NOConv, Stacked_NOBlock, Stacked_PreActivationNOBlock

from ...modules.base import LinEmbLayer
from ...modules.multi_grid.mg_base import LinearReductionLayer
from ...modules.embedding.embedder import get_embedder
from .no_helpers import get_no_layer

class MGNO_EncoderDecoder_Block(nn.Module):
  
    def __init__(self,
                 rcm: MultiRelativeCoordinateManager,
                 in_zooms: List[int],
                 in_features_list: List[int],
                 out_zooms: List[int],
                 no_zooms: List[int],
                 out_features_list: List[int],
                 no_layer_settings: dict,
                 rule = 'fc', # ">" "<"
                 block_type = 'post_layer_norm',
                 with_gamma: bool = True,
                 omit_backtransform: bool = False,
                 dropout=0,
                 embed_confs: Dict = {},
                 layer_confs: Dict = {},
                 concat_prev=False
                ) -> None: 
      
        super().__init__()
        
        self.out_zooms = out_zooms
        self.rcm=rcm
        self.layers = nn.ModuleDict()
        self.reduction_layers = nn.ModuleDict()

        
        self.concat_prev = concat_prev
        if concat_prev:
            self.zooms_concat = [out_zoom for out_zoom in out_zooms if out_zoom in in_zooms]
            self.out_features = [out_features + in_features_list[int(torch.where(torch.tensor(in_zooms)==out_zooms[k])[0][0])] if out_zooms[k] in in_zooms else out_features for k, out_features in enumerate(out_features_list)]
        else:
            self.out_features = out_features_list
            self.zooms_concat = []


        for out_idx, out_zoom in enumerate(out_zooms):

            mg_in_features = []

            layers = nn.ModuleDict()

            for in_idx, in_zoom in enumerate(in_zooms):

                zoom_diff = out_zoom - in_zoom

                if  "<" in rule and zoom_diff>0:
                    continue

                elif ">" in rule and zoom_diff<0:
                    continue

                elif "=" in rule and zoom_diff!=0:
                    continue

                if rule == ">max" and (in_zoom!=max(in_zoom) and (zoom_diff!=0)):
                    continue
                
                if rule == "<max" and (in_zoom!=max(in_zoom) and (zoom_diff!=0)):
                    continue

                in_features = in_features_list[in_idx]
                out_features = out_features_list[out_idx]
                mg_in_features.append(out_features)

                no_zoom = no_zooms[out_idx]

                if zoom_diff ==0:
                    layer = LinEmbLayer(in_features, out_features, identity_if_equal=True, layer_confs=layer_confs)
                else:
                    no_layer_type = check_get_missing_key(no_layer_settings, "no_layer_type")

                    no_layer = get_no_layer(rcm,
                                            no_layer_type,
                                            in_zoom,
                                            no_zoom,
                                            out_zoom,
                                            precompute_encode=True,
                                            precompute_decode=True,
                                            layer_settings=no_layer_settings)
                    
                    embedder = get_embedder(**embed_confs, zoom=out_zoom)
                    
                    if 'post_layer_norm' in block_type:
                        layer = NOBlock(
                                        in_features=in_features,
                                        out_features=out_features,
                                        no_layer=no_layer,
                                        with_gamma = with_gamma,
                                        OW_zero=omit_backtransform,
                                        embed_confs=embed_confs,
                                        layer_confs=layer_confs
                                        )
                    elif 'pre_layer_norm' in block_type:
                        layer = PreActivation_NOBlock(
                                    in_features=in_features,
                                    out_features=out_features,
                                    no_layer=no_layer,
                                    with_gamma = with_gamma,
                                    OW_zero=omit_backtransform,
                                    embed_confs=embed_confs,
                                    layer_confs=layer_confs
                                    )
                        
                layers[str(in_zoom)] = layer

           
            self.layers[str(out_zoom)] = layers

            reduction_layer = LinearReductionLayer(
                mg_in_features, 
                out_features,
                layer_confs)
            
            self.reduction_layers[str(out_zoom)] = reduction_layer
    

    def forward(self, x_zooms, sample_configs={}, mask_zooms=None, emb=None):

        x_zooms_out = {}
        mask_zooms_out = {}

        for out_zoom, layers in self.layers.items():

            outputs_ = []

            for in_zoom, layer in layers.items():
                x = x_zooms[int(in_zoom)]
                                
                x_out = layer(x, sample_configs=sample_configs, emb=emb)

                outputs_.append(x_out)

            x_out = self.reduction_layers[out_zoom](outputs_, emb=emb)

            x_zooms_out[int(out_zoom)] = x_out

        for zoom in self.zooms_concat:
            x_zooms_out[zoom] = torch.concat((x_zooms_out[zoom], x_zooms[zoom]),dim=-1)

        return x_zooms_out

class MGNO_StackedEncoderDecoder_Block(nn.Module):
  
    def __init__(self,
                 rcm: MultiRelativeCoordinateManager,
                 in_zooms: List[int],
                 in_features_list: List[int],
                 out_zooms: List[int],
                 no_zoom: int,
                 out_features_list: List[int],
                 no_layer_settings: dict,
                 no_zoom_step: int = 1,
                 concat_features = 1,
                 block_type = 'post_layer_norm',
                 with_gamma = False,
                 p_dropout=0,
                 concat_prev=False,
                 layer_confs={},
                 embed_confs={},
                ) -> None: 
      
        super().__init__()

        self.concat_prev = concat_prev
        self.out_zooms = out_zooms
        self.rcm=rcm

        no_layer_type = check_get_missing_key(no_layer_settings, "no_layer_type")
        no_layers = nn.ModuleList()

        self.out_features = out_features_list
        
        self.in_zooms = in_zooms
        


        for input_zoom in range(int(torch.tensor(in_zooms).max()), no_zoom + 1 - no_zoom_step, (-1) * no_zoom_step):

            no_zoom_k = input_zoom + (-1) * no_zoom_step

            no_layer = get_no_layer(rcm,
                                    no_layer_type,
                                    input_zoom,
                                    no_zoom_k,
                                    input_zoom,
                                    precompute_encode=True,
                                    precompute_decode=True,
                                    layer_settings=no_layer_settings)

            no_layers.append(no_layer)

                
        no_conv_layer = Stacked_NOConv(
            in_zooms,
            in_features_list, 
            out_features_list,
            no_layers,
            out_zooms,
            layer_confs=layer_confs,
            concat_features=concat_features
            )

        if block_type == 'post_layer_norm':
            self.layer = Stacked_NOBlock(no_conv_layer,
                                        with_gamma=with_gamma,
                                        p_dropout=p_dropout,
                                        layer_confs=layer_confs,
                                        embed_confs=embed_confs)
        
        elif block_type == 'pre_layer_norm':
            self.layer = Stacked_PreActivationNOBlock(no_conv_layer,
                                        with_gamma=with_gamma,
                                        p_dropout=p_dropout,
                                        layer_confs=layer_confs,
                                        embed_confs=embed_confs)
        
        if concat_prev:
            self.zooms_concat = [out_zoom for out_zoom in out_zooms if out_zoom in in_zooms]
            self.out_features = [out_features + in_features_list[int(torch.where(torch.tensor(in_zooms)==out_zooms[k])[0][0])] if out_zooms[k] in in_zooms else out_features for k, out_features in enumerate(out_features_list)]
        else:
            self.zooms_concat = []

    def forward(self, x_zooms, sample_configs={}, mask_zooms=None, emb=None):
        
        x_zooms_out = self.layer(x_zooms, sample_configs=sample_configs, mask_zooms=mask_zooms, emb=emb)

        for zoom in self.zooms_concat:
            x_zooms_out[zoom] = torch.concat((x_zooms_out[zoom], x_zooms[zoom]), dim=-1)

        return x_zooms_out