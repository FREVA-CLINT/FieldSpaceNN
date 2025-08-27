from typing import List, Dict,Tuple,Optional
import warnings

from omegaconf import ListConfig

import torch
import torch.nn as nn

from ..transformer.transformer_base import TransformerBlock
from ..base import LinEmbLayer, MLP_fac, get_layer

from ..grids.grid_layer import GridLayer

from ...modules.embedding.embedder import get_embedder
from .mg_attention import MultiZoomSelfAttention,MultiZoomFieldAttention
from .mg_base import NHConv, ResNHConv, FieldLayer#,get_weight_matrix,get_einsum_subscripts

class MG_SingleBlock(nn.Module):
  
    def __init__(self,
                 grid_layers: Dict[str,GridLayer],
                 in_zooms: List[int],
                 layer_settings: Dict,
                 in_features_list: List[int],
                 out_features_list: List[int],
                 zooms = None,
                 layer_confs = {},
                 layer_confs_emb={},
                 use_mask=False
                ) -> None: 
      
        super().__init__()

        self.in_zooms = in_zooms
        self.out_features = out_features_list
        self.out_zooms = in_zooms

        self.blocks = nn.ModuleDict()
        self.use_mask = use_mask

        zooms = in_zooms if zooms is None else zooms 

        if not (len(zooms) == len(out_features_list)):
            warnings.warn(
                f"Length mismatch: zooms({len(zooms)}), "
                f"in_features_list({len(in_features_list)}), "
                f"out_features_list({len(out_features_list)})"
            )

        k = 0
        for zoom, in_features, out_features in zip(zooms, in_features_list, out_features_list):
            
            type = layer_settings.get('type','TransformerBlock')

            embedders = get_embedder(**layer_settings.get('embed_confs', {}), grid_layers=grid_layers,zoom=zoom)

            if type == 'TransformerBlock':
                seq_length = layer_settings.get('seq_lengths', [10]*len(in_zooms))[k]
                zoom_block = max([zoom - seq_length,0])
                
                block = TransformerBlock(
                            in_features,
                            out_features,
                            layer_settings['blocks'],
                            seq_lengths=seq_length,
                            num_heads=layer_settings.get('num_heads', 2),
                            n_head_channels=layer_settings.get('n_head_channels', None),
                            att_dims=layer_settings.get('att_dims', None),
                            mlp_mult=layer_settings.get('mlp_mult', 1),
                            dropout=layer_settings.get('dropout', 1),
                            spatial_dim_count=layer_settings.get('spatial_dim_count', 1),
                            embedders=embedders,
                            layer_confs=layer_confs,
                            grid_layer=grid_layers[str(zoom_block)],
                            layer_confs_emb=layer_confs_emb
                        )
                    
            elif type == 'nh_conv':
                ranks_spatial = layer_settings.get('ranks_spatial', [])

                block = NHConv(
                    grid_layers[str(zoom)],
                    in_features,
                    out_features,
                    ranks_spatial=ranks_spatial,
                    layer_confs=layer_confs
                )
            
            elif type == 'res_nh_conv':
                ranks_spatial = layer_settings.get('ranks_spatial', [])

                block = ResNHConv(
                    grid_layers[str(zoom)],
                    in_features,
                    out_features,
                    ranks_spatial=ranks_spatial,
                    layer_confs=layer_confs
                )

            elif type == 'linear':
                ranks_spatial = layer_settings.get('ranks_spatial', [])

                block = LinEmbLayer(
                    in_features,
                    out_features, 
                    layer_norm=False, 
                    identity_if_equal=False,
                    embedder=embedders,
                    layer_confs=layer_confs,
                    layer_confs_emb=layer_confs_emb)
                
            elif type == 'mlp':

                block = MLP_fac(
                    in_features,
                    out_features, 
                    mult=layer_settings.get('mlp_mult', 1), 
                    dropout=layer_settings.get('dropout', 0), 
                    layer_confs=layer_confs)

            self.blocks[str(zoom)] = block
        

    def forward(self, x_zooms: Dict, sample_configs={},  emb=None, mask_zooms={}, **kwargs):

        for zoom, block in self.blocks.items():
            x = x_zooms[int(zoom)]

            x = block(x, emb=emb, sample_configs=sample_configs[int(zoom)], mask=mask_zooms[int(zoom)] if self.use_mask else None)

            x_zooms[int(zoom)] = x

        return x_zooms
    

class MG_MultiBlock(nn.Module):
  
    def __init__(self, 
                 grid_layers: Dict[str, GridLayer],
                 in_zooms: List[int],
                 out_zooms: List[int],
                 layer_settings: Dict,
                 in_features: int,
                 out_features: int,
                 q_zooms:List|int = -1,
                 kv_zooms:List|int = -1,
                 layer_confs={},
                 layer_confs_emb={},
                 use_mask = False,
                 type=''
                 ) -> None:
        super().__init__()
        
        self.out_features = [out_features]*len(out_zooms)
        self.out_zooms = out_zooms
        self.use_mask = use_mask

        self.zero_zooms_gen = nn.ParameterDict()

        for zoom in out_zooms:
            if zoom not in in_zooms:
                self.zero_zooms_gen[str(zoom)] = nn.Parameter(torch.zeros((1,1,1,12*4**zoom,1)), requires_grad=False)


        self.blocks = nn.ModuleDict()

        if isinstance(in_features, (List,ListConfig)) and len(set(in_features)) > 1:
            raise ValueError("features of levels should be the same.")
        elif isinstance(in_features, (int)):
            in_features = in_features
        else:
            in_features = in_features[0]

        if not isinstance(q_zooms, (List,ListConfig)) and (q_zooms == -1):
            q_zooms = in_zooms
        
        if not isinstance(kv_zooms,(List,ListConfig)) and (kv_zooms == -1):
            kv_zooms = in_zooms

        
        compression_zooms = layer_settings.get("compression_zooms",{})

        zooms_embedders = torch.tensor(q_zooms + kv_zooms + list(compression_zooms.values())).unique() 


        embedders = {}

        for k, zoom in enumerate(zooms_embedders):
            #if zoom not in in_zooms:
            #    raise ValueError(f"Zoom level {zoom} at index {k} of q_zooms not found in in_zooms")
            
            embedder = get_embedder(**layer_settings.get('embed_confs', {}), grid_layers=grid_layers, zoom=int(zoom))
            embedders[str(int(zoom))] = embedder
            
        
        for k, zoom in enumerate(kv_zooms):
            if zoom not in in_zooms:
                raise ValueError(f"Zoom level {zoom} at index {k} of kv_zooms not found in in_zooms")
            
        att_zoom = (min(q_zooms + kv_zooms))  
        att_zoom = min([layer_settings.get("att_zoom",att_zoom), att_zoom])
        grid_layer = grid_layers[str(att_zoom)]

        
        if type=='field_att':
            block = MultiZoomFieldAttention(grid_layer,
                                in_features,
                                out_features,
                                q_zooms,
                                kv_zooms,
                                mult = layer_settings.get("mlp_mult",1),
                                num_heads = layer_settings.get("num_heads",1),
                                n_head_channels= layer_settings.get("n_head_channels",16),
                                share_zoom_proj = layer_settings.get("share_zoom_proj",False),
                                share_zoom_proj_qkv = layer_settings.get("share_zoom_proj_qkv",False),
                                with_nh= layer_settings.get("with_nh",True),
                                var_att= layer_settings.get("var_att",False),
                                rank=layer_settings.get("rank",16),
                                contract_zooms=layer_settings.get("contract_zooms",True),
                                contract_channels=layer_settings.get("contract_channels",True),
                                embedders=embedders,
                                layer_confs=layer_confs,
                                layer_confs_emb=layer_confs_emb
                                )
        else:
            block = MultiZoomSelfAttention(
                        grid_layers,
                        att_zoom,
                        in_features,
                        out_features,
                        q_zooms,
                        kv_zooms,
                        mult = layer_settings.get("mlp_mult",1),
                        num_heads = layer_settings.get("num_heads",1),
                        n_head_channels = layer_settings.get("n_head_channels",None),
                        compression_dims_kv =layer_settings.get("compression_dims_kv",{}),
                        compression_zooms=layer_settings.get("compression_zooms",{}),
                        att_dim=layer_settings.get('att_dim', None),
                        with_nh= layer_settings.get("with_nh",True),
                        var_att= layer_settings.get("var_att",False),
                        common_kv = layer_settings.get("common_kv",False),
                        common_q = layer_settings.get("common_q",False),
                        embedders=embedders,
                        layer_confs=layer_confs,
                        layer_confs_emb=layer_confs_emb
                        )


        self.block = block

    def generate_zoom(self, in_shape, zoom, zoom_patch_sample=-1, n_past_ts=0, n_future_ts=0, **kwargs):
        nt = n_past_ts + n_future_ts + 1

        x_zoom = self.zero_zooms_gen[str(zoom)]

        if zoom_patch_sample > -1:
            x_zoom = x_zoom.view(1,1,1,12*4**zoom_patch_sample,-1,1)[:,:,:,0]

        return x_zoom.expand(in_shape[0],in_shape[1],nt,-1,in_shape[-1])


    def forward(self, x_zooms, emb=None, mask_zooms={}, sample_configs={}):

        for zoom in self.out_zooms:
            if zoom not in x_zooms.keys():
                x = self.generate_zoom(list(x_zooms.values())[0].shape, zoom, **sample_configs[zoom])
                x_zooms[zoom] = x
    
        x_zooms = self.block(x_zooms, emb=emb, mask_zooms=mask_zooms if self.use_mask else {}, sample_configs=sample_configs)

        x_zooms_out = {}
        for zoom in self.out_zooms:
            x_zooms_out[zoom] = x_zooms[zoom]

        return x_zooms_out