import math
import copy
from typing import List,Dict

from ..utils.helpers import check_get

import torch
import torch.nn as nn

from ..modules.grids.grid_layer import GridLayer
from .factorization import SpatiaFacLayer


class IdentityLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, **kwargs):
        return x

class LinEmbLayer(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 layer_norm = False,
                 identity_if_equal = False,
                 layer_confs: dict={},
                 layer_confs_emb: dict={},
                 embedder=None,
                 spatial_dim_count=1
                ) -> None: 
         
        super().__init__()

        self.embedder = embedder
        self.spatial_dim_count = spatial_dim_count

        if isinstance(out_features, list):
            out_features_ = int(torch.tensor(out_features).prod())
        else:
            out_features_ = out_features

        if self.embedder is not None:
            self.embedding_layer = get_layer(self.embedder.get_out_channels, [2, out_features_], layer_confs=layer_confs_emb)
         
            self.forward_fcn = self.forward_w_embedding
        else:
            self.forward_fcn = self.forward_wo_embedding

        if layer_norm:
            self.layer_norm = nn.LayerNorm(out_features_, elementwise_affine=True)
        else:
            self.layer_norm = nn.Identity()

        if identity_if_equal and in_features==out_features:
            self.layer = IdentityLayer()
        else:
            self.layer = get_layer(in_features, out_features, layer_confs=layer_confs)

    def forward_w_embedding(self, x, emb=None, sample_configs={}):
        
        x = self.forward_wo_embedding(x, emb=emb, sample_configs=sample_configs)

        x_shape = x.shape

        emb_ = self.embedder(emb, sample_configs)
        scale, shift = self.embedding_layer(emb_, sample_configs=sample_configs, emb=emb).chunk(2, dim=-2)

        n = scale.shape[1]
        scale = scale.view(*scale.shape[:3], -1, x_shape[-1])
        shift = shift.view(*shift.shape[:3], -1, x_shape[-1])

        x = x * (scale + 1) + shift

        return x

    def forward_wo_embedding(self, x, emb=None, sample_configs={}):

        x = self.layer(x, emb=emb, sample_configs=sample_configs)
        x = x.view(*x.shape[:4 + self.spatial_dim_count-1],-1)
        x = self.layer_norm(x)

        return x

    def forward(self, x, emb=None, sample_configs={},**kwargs):
        return self.forward_fcn(x, emb=emb, sample_configs=sample_configs)


    
class MLP_fac(nn.Module):
  
    def __init__(self,
                 in_features, 
                 out_features,
                 mult=1,
                 dropout=0,
                 layer_confs: Dict={},
                 gamma=False
                ) -> None: 
      
        super().__init__() 
        
        self.layer1 = get_layer(in_features, int(in_features * mult), layer_confs=layer_confs, bias=True)
        self.layer2 = get_layer(int(in_features * mult), out_features, layer_confs=layer_confs, bias=True)
        self.dropout = nn.Dropout(p=dropout) if dropout>0 else nn.Identity()
        self.activation = nn.SiLU()

        if gamma:
            self.gamma = torch.nn.Parameter(torch.ones(out_features) * 1E-6)
            self.rtn_fcn = self.rtn_w_gamma
        else:
            self.rtn_fcn = self.rtn

    def rtn_w_gamma(self, x):
        return x * self.gamma
    
    def rtn(self, x):
        return x

    def forward(self, x, emb=None, **kwargs):
        
        x = self.layer1(x, emb=emb)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer2(x, emb=emb)

        return self.rtn_fcn(x)


def get_layer(
        in_features,
        out_features,
        layer_confs: Dict={},
        **kwargs
        ):  

    layer_confs = copy.deepcopy(layer_confs)
        
    rank_feat = check_get([layer_confs, kwargs, {'rank_feat': None}], 'rank_feat')
    rank_vars = check_get([layer_confs, kwargs, {'rank_vars': None}], 'rank_vars')
    n_groups = check_get([layer_confs, kwargs, {'n_groups': 1}], 'n_groups')
    rank_channel = check_get([layer_confs, kwargs, {'rank_channel': None}], 'rank_channel')
    bias = check_get([layer_confs, kwargs, {'bias': False}], 'bias')

    if not rank_feat and not rank_vars and rank_channel is None and n_groups==1:
        layer = LinearLayer(
                in_features,
                out_features,
                bias=bias
                )
    else:
    
        """
        layer = SpatiaFacLayer(
            in_features,
            out_features,
            ranks_spatial=ranks_spatial,
            dims_spatial=dims_spatial,
            rank_feat=rank_feat,
            rank_vars=rank_vars,
            rank_channel=rank_channel,
            base=base,
            n_groups=n_groups,
            sum_n_zooms=sum_n_zooms,
            bias=bias)
        """
        layer = SpatiaFacLayer(
            in_features,
            out_features,
            **layer_confs)

    return layer


class LinearLayer(nn.Module):
    def __init__(self, 
                 in_features: int|List, 
                 out_features: int|List,
                 bias=False,
                 **kwargs):

        super().__init__()

        if isinstance(in_features,int):
            in_features = [in_features]

        if isinstance(out_features,int):
            out_features = [out_features]

       # self.out_features = int(torch.tensor(out_features).prod())

        self.layer = nn.Linear(int(torch.tensor(in_features).prod()), int(torch.tensor(out_features).prod()), bias=bias)

        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x, **kwargs):
        x_dims = x.shape[:-len(self.in_features)]

        x = x.view(*x_dims, *self.in_features)
        x = self.layer(x)
        x = x.view(*x_dims, *self.out_features)
        
        return x




