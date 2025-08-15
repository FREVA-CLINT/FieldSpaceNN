from typing import List,Dict
from einops import rearrange
import copy

import torch
import torch.nn as nn

from ..base import get_layer, IdentityLayer, LinEmbLayer, MLP_fac

from ...modules.grids.grid_layer import GridLayer
from ...modules.transformer.transformer_base import SelfAttention

from ...modules.embedding.embedder import EmbedderSequential
from ..grids.grid_utils import get_matching_time_patch, insert_matching_time_patch

    

class MultiZoomSelfAttention(nn.Module):
  
    def __init__(self, 
                 grid_layer: GridLayer,
                 in_features: int,
                 out_features: int,
                 q_zooms: int,
                 kv_zooms: int,
                 mult: int=1,
                 dropout: float=0.0,
                 num_heads: int=1,
                 att_dim=None,
                 n_head_channels: int=None, 
                 embedders: Dict[str, EmbedderSequential] = {},
                 compression_dims_kv: Dict[int, int]= {},
                 with_nh = True,
                 var_att = False,
                 common_kv = False,
                 common_q = False,
                 common_out = False,
                 layer_confs: Dict = {},
                 layer_confs_emb= {}) -> None: 
        
        super().__init__()
        
        layer_confs_kv = copy.deepcopy(layer_confs)
        layer_confs_kv['bias'] = True

        self.with_nh = with_nh

        self.emb_layers = nn.ModuleDict()
        self.mlp_emb_layers = nn.ModuleDict()
        self.q_layers = nn.ModuleDict()
        self.kv_layers = nn.ModuleDict()
        self.mlps = nn.ModuleDict()
        self.out_layers = nn.ModuleDict()
       # self.gammas = nn.ParameterDict()

        att_dim = out_features if att_dim is None else att_dim

        for in_zoom in embedders.keys():
            self.emb_layers[in_zoom] = LinEmbLayer(in_features, [in_features], layer_confs=layer_confs, identity_if_equal=True, embedder=embedders[in_zoom], layer_norm=True, layer_confs_emb=layer_confs_emb) 

        for q_zoom in q_zooms:
            self.mlp_emb_layers[str(q_zoom)] = LinEmbLayer(out_features, out_features, layer_confs=layer_confs, identity_if_equal=True, embedder=embedders[str(q_zoom)], layer_norm=True, layer_confs_emb=layer_confs_emb)
            self.mlps[str(q_zoom)] = MLP_fac(out_features, out_features, mult, dropout, layer_confs=layer_confs, gamma=True) 

            self.q_layers[str(q_zoom)] = get_layer(in_features, out_features, layer_confs=layer_confs) if not common_q else IdentityLayer()
            self.out_layers[str(q_zoom)] = get_layer(in_features, out_features, layer_confs=layer_confs) if not common_out else IdentityLayer()

        self.omit_mask_zooms = []
        for kv_zoom in kv_zooms:
            in_f = out_f = []

            if kv_zoom in compression_dims_kv.keys():
                in_f = [4 for _ in compression_dims_kv[kv_zoom]]
                out_f = [4 if d==-1 else d for d in compression_dims_kv[kv_zoom]]

                self.omit_mask_zooms.append(kv_zoom)

            if with_nh:
                nh_dim = grid_layer.adjc.shape[1] if with_nh else None
                in_features_kv = [nh_dim, *in_f, in_features]
                out_features_kv = [nh_dim, *out_f, 2*out_features]
                layer_confs_ = layer_confs.copy()
                layer_confs_['skip_dims'] = [True, *[False]*len(out_f), False]
            else:
                layer_confs_ = layer_confs
                in_features_kv = [*in_f, in_features]
                out_features_kv = [*out_f, 2*out_features]

            self.kv_layers[str(kv_zoom)] = get_layer(in_features_kv, out_features_kv, layer_confs=layer_confs_,bias=True) if not common_kv else IdentityLayer()

   
        self.common_q_layer = get_layer(in_features, in_features, bias=False, layer_confs=layer_confs) if common_q else IdentityLayer()
        self.common_kv_layer = get_layer(in_features, [2*in_features], bias=True, layer_confs=layer_confs) if common_kv else IdentityLayer()

        num_heads = num_heads if not n_head_channels else in_features // n_head_channels
        self.attention = SelfAttention(in_features, in_features, num_heads=num_heads, dropout=dropout, cross=True, qkv_proj=False)

        self.grid_layer = grid_layer
        self.max_zoom = int(max(list(embedders.keys())))

        assert out_features==in_features,"Module does not support differen in and out features"

        if not var_att:
            self.pattern = 'b v t s n c -> (b v t s) (n) c'
            self.kv_pattern = 'b v t s n kv c -> (b v t s) (n) (kv c)'
            self.nh_mask_pattern = 'b v t s n 1 -> (b v t s) 1 1 n'
            self.reverse_pattern = '(b v t s) n c -> b v t s n c'
        else:
            self.pattern = 'b v t s n c -> (b t s) (v n) c'
            self.kv_pattern = 'b v t s n kv c -> (b t s) (v n) (kv c)'
            self.nh_mask_pattern = 'b v t s n 1 -> (b t s) 1 1 (v n)'
            self.reverse_pattern = '(b t s) (v n) c -> b v t s n c'

    def forward(self, x_zooms, mask_zooms=None, emb=None, sample_configs={}):        

        zoom_att = self.grid_layer.zoom

        q = []
        kv = []
        mask = []
        n_p = []
        x_zooms_emb = {}

        for zoom, emb_layer in self.emb_layers.items():
            x_zooms_emb[int(zoom)] = emb_layer(x_zooms[int(zoom)], emb=emb, sample_configs=sample_configs[int(zoom)])

        
        for zoom, q_layer in self.q_layers.items():
            q_ = get_matching_time_patch(x_zooms_emb[int(zoom)], int(zoom), self.max_zoom, sample_configs)
            n = q_.shape[3] // 4**(int(zoom)-zoom_att)

            q_ = q_layer(q_, emb=emb, sample_configs=sample_configs[int(zoom)])
            q_ = q_.view(*q_.shape[:3], n, -1 ,q_.shape[-1])
            n_p.append(q_.shape[-2])
            q.append(q_)

        for zoom, kv_layer in self.kv_layers.items():
            kv_, mask_ = self.grid_layer.get_nh(x_zooms_emb[int(zoom)], **sample_configs[int(zoom)], with_nh=self.with_nh, mask=mask_zooms[int(zoom)] if len(mask_zooms)>0 else None)
            kv_ = get_matching_time_patch(kv_, int(zoom), self.max_zoom, sample_configs)
            
            n = kv_.shape[3]
            kv_ = kv_layer(kv_, emb=emb, sample_configs=sample_configs[int(zoom)])
            kv_ = kv_.reshape(*kv_.shape[:3],n,-1,kv_.shape[-1])

            if mask_ is not None and int(zoom) not in self.omit_mask_zooms:
                mask_ = get_matching_time_patch(mask_, int(zoom), self.max_zoom, sample_configs)
            else:
                mask_ = torch.zeros_like(kv_[...,[0]], dtype=torch.bool, device=kv_.device)

            kv.append(kv_)
            mask.append(mask_)

        kv = torch.concat(kv, dim=-2)
        q = torch.concat(q, dim=-2)

        q = self.common_q_layer(q, sample_configs=sample_configs[int(zoom)], emb=emb)
        kv_shape = kv.shape[:-1]
        kv = self.common_kv_layer(kv, sample_configs=sample_configs[int(zoom)], emb=emb).view(*kv_shape,2,-1)

        if mask[0] is not None:
            mask = torch.concat(mask, dim=-2)
            mask = rearrange(mask, self.nh_mask_pattern)
        else:
            mask = None

        b, v, t, s, n, c = q.shape

        q = rearrange(q, self.pattern)
        kv = rearrange(kv, self.kv_pattern)

        q = self.attention(q, kv, mask=mask)

        q = rearrange(q, self.reverse_pattern, b=b, t=t, s=s, n=n, v=v, c=c) 

        q = q.split(n_p, dim=-2)

        for k, zoom in enumerate(self.q_layers.keys()):
            
            x_q = insert_matching_time_patch(x_zooms[int(zoom)], q[k].reshape(*q[k].shape[:3],-1,q[k].shape[-1]), int(zoom), self.max_zoom, sample_configs)

            x_zooms[int(zoom)] = x_zooms[int(zoom)] + x_q #q[k].reshape(x_zooms[int(zoom)].shape)
        
            x_mlp = self.mlp_emb_layers[zoom](x_zooms[int(zoom)], emb=emb, sample_configs=sample_configs[int(zoom)])

            x_zooms[int(zoom)] = x_zooms[int(zoom)] + self.mlps[zoom](x_mlp, emb=emb, sample_configs=sample_configs[int(zoom)])

            x_zooms[int(zoom)] = self.out_layers[zoom](x_zooms[int(zoom)], emb=emb, sample_configs=sample_configs[int(zoom)])


        return x_zooms