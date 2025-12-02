from typing import List, Dict,Tuple,Optional
import warnings
import copy

from omegaconf import ListConfig

import torch
import torch.nn as nn

from ..transformer.transformer_base import TransformerBlock
from ..base import LinEmbLayer, MLP_fac, get_layer

from ..grids.grid_layer import GridLayer

from ...modules.embedding.embedder import get_embedder
from .mg_attention import MultiZoomSelfAttention,MultiFieldAttention
from .mg_base import Conv, ResConv, refine_zoom, coarsen_zoom

def invert_dict(d):
    inverted_d = {}
    for key, value in d.items():
        inverted_d[value] = key
    return inverted_d

class MG_SingleBlock(nn.Module):
  
    def __init__(self,
                 grid_layers: Dict[str,GridLayer],
                 in_zooms: List[int],
                 layer_settings: Dict,
                 in_features_list: List[int],
                 out_features_list: List[int],
                 zooms: List[int]=None,
                 layer_confs = {},
                 layer_confs_emb={},
                 use_mask=False,
                 n_head_channels=16
                ) -> None: 
      
        super().__init__()

        zooms = in_zooms if zooms is None else zooms

        self.out_features = out_features_list
        self.out_zooms = in_zooms

        self.blocks = nn.ModuleDict()
        self.use_mask = use_mask


        if not (len(zooms) == len(out_features_list)):
            warnings.warn(
                f"Length mismatch: zooms({len(zooms)}), "
                f"in_features_list({len(in_features_list)}), "
                f"out_features_list({len(out_features_list)})"
            )

        k = 0
        for k, (zoom, out_features) in enumerate(zip(zooms, out_features_list)):
            
            in_features = in_features_list[k] if k <= len(in_features_list) -1 else None

            type = layer_settings.get('type','TransformerBlock')

            embedders = get_embedder(**layer_settings.get('embed_confs', {}), grid_layers=grid_layers,zoom=zoom)

            if type == 'TransformerBlock':
                seq_length = layer_settings.get('seq_lengths', [10]*len(zooms))[k]
                zoom_block = max([zoom - seq_length,0])
                
                block = TransformerBlock(
                            in_features,
                            out_features,
                            layer_settings['blocks'],
                            seq_lengths=seq_length,
                            num_heads=layer_settings.get('num_heads', None),
                            n_head_channels=layer_settings.get('n_head_channels', n_head_channels),
                            att_dims=layer_settings.get('att_dims', None),
                            mlp_mult=layer_settings.get('mlp_mult', 1),
                            dropout=layer_settings.get('dropout', 1),
                            spatial_dim_count=layer_settings.get('spatial_dim_count', 1),
                            embedders=embedders,
                            layer_confs=layer_confs,
                            grid_layer=grid_layers[str(zoom_block)],
                            layer_confs_emb=layer_confs_emb
                        )
                    
            elif type == 'conv':
                ranks_spatial = layer_settings.get('ranks_spatial', [])

                block = Conv(
                    grid_layers[str(zoom)],
                    in_features,
                    out_features,
                    ranks_spatial=ranks_spatial,
                    layer_confs=layer_confs
                    )
                
            elif type == 'res_conv':
                ranks_spatial = layer_settings.get('ranks_spatial', [])

                block = ResConv(
                    grid_layers[str(zoom)],
                    in_features,
                    out_features,
                    ranks_spatial=ranks_spatial,
                    layer_confs=layer_confs,
                    layer_confs_emb=layer_confs_emb,
                    with_gamma=layer_settings.get('with_gamma',False),
                    embedder=embedders
                )

            elif type == 'linear':
                ranks_spatial = layer_settings.get('ranks_spatial', [])

                block = LinEmbLayer(
                    in_features,
                    out_features, 
                    layer_norm=False, 
                    identity_if_equal=layer_settings.get('identity_if_equal', False),
                    embedder=embedders,
                    layer_confs=layer_confs,
                    layer_confs_emb=layer_confs_emb)
                
                self.out_features[k] = block.out_features
                
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
            x = x_zooms[int(zoom)] if int(zoom) in x_zooms.keys() else 0

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
                 dropout:float = 0,
                 layer_confs={},
                 layer_confs_emb={},
                 use_mask = False,
                 n_head_channels=16,
                 refine_zooms= {},
                 shift = None,
                 rev_shift = True,
                 multi_shift = False
                 ) -> None:
        super().__init__()
        
        self.out_features = [out_features]*len(out_zooms)
        self.out_zooms = copy.deepcopy(out_zooms)
        in_zooms = copy.deepcopy(in_zooms)
        self.use_mask = use_mask


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

        for k,zoom in enumerate(q_zooms):
            if zoom in refine_zooms.keys():
                q_zooms[k] = refine_zooms[zoom]
        
        for k,zoom in enumerate(kv_zooms):
            if zoom in refine_zooms.keys():
                kv_zooms[k] = refine_zooms[zoom]

        for k,zoom in enumerate(in_zooms):
            if zoom in refine_zooms.keys():
                in_zooms[k] = refine_zooms[zoom]

        for k, zoom in enumerate(kv_zooms):
            if zoom not in in_zooms:
                raise ValueError(f"Zoom level {zoom} at index {k} of kv_zooms not found in in_zooms")
        
        self.qkv_zooms = torch.tensor(q_zooms + kv_zooms).unique().tolist()

        att_zoom = (min(q_zooms + kv_zooms))  
        att_zoom = min([layer_settings.get("att_zoom",att_zoom), att_zoom])

        field_zoom = layer_settings.get("field_zoom", att_zoom)

        if (min(q_zooms + kv_zooms)) < field_zoom:
            raise ValueError(f"Zoom level {min(q_zooms + kv_zooms)} need to be refined. please indicate refine_zooms={refine_zooms}")


        embedder = get_embedder(**layer_settings.get('embed_confs', {}), grid_layers=grid_layers, zoom=int(field_zoom))
        residual_embedder = get_embedder(**layer_settings.get('residual_embed_confs', {}), grid_layers=grid_layers, zoom=int(field_zoom))

        block = MultiFieldAttention(
                    grid_layers[str(field_zoom)],
                    grid_layers[str(att_zoom)],
                    q_zooms,
                    kv_zooms,
                    mult = layer_settings.get("mlp_mult",1),
                    att_dim = layer_settings.get("att_dim", None),
                    n_head_channels = layer_settings.get("n_head_channels",n_head_channels),
                    with_nh_field_mlp = layer_settings.get("with_nh_field_mlp", False),
                    with_nh_field = layer_settings.get("with_nh_field", True),
                    with_nh_att = layer_settings.get("with_nh_att", False),
                    with_var_att= layer_settings.get("with_var_att", False),
                    with_time_att= layer_settings.get("with_time_att", False),
                    time_seq_len= layer_settings.get("time_seq_len", 1),
                    with_nh_post_att = layer_settings.get("with_nh_post_att", False),
                    with_mlp_embedder=layer_settings.get("with_mlp_embedder", True),
                    dropout=dropout,
                    embedder=embedder,
                    residual_embedder=residual_embedder,
                    layer_confs=layer_confs,
                    layer_confs_emb=layer_confs_emb,
                    residual_learned=layer_settings.get("residual_learned", False),
                    update=layer_settings.get("update", 'shift'),
                    double_skip = layer_settings.get("double_skip", False),
                    layer_norm = layer_settings.get("layer_norm", True)
                    )

        self.grid_layers = grid_layers
        self.multi_shift = multi_shift
        self.field_zoom = field_zoom

        if shift and rev_shift:
            self.fwd_fcn = self.forward_with_rev_shift
        elif shift:
            self.fwd_fcn = self.forward_with_shift
        else:
            self.fwd_fcn = self.forward_
        
        self.window_shift = shift
        self.refine_zooms = refine_zooms
        self.coarse_zooms = invert_dict(refine_zooms)

        self.block = block

    def generate_zoom(self, in_shape, zoom, device, zoom_patch_sample=-1, n_past_ts=0, n_future_ts=0, **kwargs):
        nt = n_past_ts + n_future_ts + 1

        x_zoom = self.init_missing_zooms((1,1,1,12*4**zoom,1)).to(device)

        if zoom_patch_sample > -1:
            x_zoom = x_zoom.view(1,1,1,12*4**zoom_patch_sample,-1,1)[:,:,:,0]

        return x_zoom.expand(in_shape[0],in_shape[1],nt,-1,in_shape[-1])


    def forward_with_rev_shift(self, x_zooms, emb=None, mask_zooms={}, sample_configs={}):
        for in_zoom, out_zoom in self.refine_zooms.items():
            x_zooms[out_zoom] = refine_zoom(x_zooms[in_zoom], in_zoom, out_zoom)
        
        if self.window_shift is not None:
            for zoom in self.qkv_zooms:
                grid_layer = self.grid_layers[str(self.field_zoom + 1)] if not self.multi_shift else self.grid_layers[str(zoom)]
                x_zooms[zoom] = grid_layer.apply_shift(x_zooms[zoom], self.window_shift, **sample_configs[zoom])[0]

        x_zooms = self.block(x_zooms, emb=emb, mask_zooms=mask_zooms if self.use_mask else {}, sample_configs=sample_configs)

        if self.window_shift is not None:
            for zoom in self.qkv_zooms:
                grid_layer = self.grid_layers[str(self.field_zoom + 1)] if not self.multi_shift else self.grid_layers[str(zoom)]
                x_zooms[zoom] = grid_layer.apply_shift(x_zooms[zoom], self.window_shift, **sample_configs[zoom], reverse=True)[0]

        for in_zoom, out_zoom in self.coarse_zooms.items():
            x_zooms[out_zoom] = coarsen_zoom(x_zooms[in_zoom], in_zoom, out_zoom)

        return x_zooms


    def forward_with_shift(self, x_zooms, emb=None, mask_zooms={}, sample_configs={}):
        for in_zoom, out_zoom in self.refine_zooms.items():
            x_zooms[out_zoom] = refine_zoom(x_zooms[in_zoom], in_zoom, out_zoom)
        
        x_zooms_shift = x_zooms.copy()

        for zoom in self.qkv_zooms:
            grid_layer = self.grid_layers[str(self.field_zoom + 1)] if not self.multi_shift else self.grid_layers[str(zoom)]
            x_zooms_shift[zoom] = grid_layer.apply_shift(x_zooms_shift[zoom], self.window_shift, **sample_configs[zoom])[0]

        x_zooms = self.block(x_zooms_shift, emb=emb, mask_zooms=mask_zooms if self.use_mask else {}, sample_configs=sample_configs, x_zoom_res=x_zooms)

        for in_zoom, out_zoom in self.coarse_zooms.items():
            x_zooms[out_zoom] = coarsen_zoom(x_zooms[in_zoom], in_zoom, out_zoom)

        return x_zooms


    def forward_(self, x_zooms, emb=None, mask_zooms={}, sample_configs={}):
        for in_zoom, out_zoom in self.refine_zooms.items():
            x_zooms[out_zoom] = refine_zoom(x_zooms[in_zoom], in_zoom, out_zoom)
        
        x_zooms_shift = x_zooms.copy()

        x_zooms = self.block(x_zooms_shift, emb=emb, mask_zooms=mask_zooms if self.use_mask else {}, sample_configs=sample_configs, x_zoom_res=x_zooms)

        for in_zoom, out_zoom in self.coarse_zooms.items():
            x_zooms[out_zoom] = coarsen_zoom(x_zooms[in_zoom], in_zoom, out_zoom)

        return x_zooms
    

    def forward(self, x_zooms, emb=None, mask_zooms={}, sample_configs={}):
        
        x_zooms = self.fwd_fcn(x_zooms, emb=emb, mask_zooms=mask_zooms, sample_configs=sample_configs)
        
        x_zooms_out = {}
        for zoom in self.out_zooms:
            x_zooms_out[zoom] = x_zooms[zoom]

        return x_zooms_out
    