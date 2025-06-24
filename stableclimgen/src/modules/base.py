import math
import copy
from typing import List,Dict

from ..utils.helpers import check_get

import torch
import torch.nn as nn

from ..modules.grids.grid_layer import GridLayer
from .factorization import SpatiaFacLayer


from ..modules.embedding.embedder import EmbedderSequential


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
                 embedder: EmbedderSequential=None,
                ) -> None: 
         
        super().__init__()

        self.embedder = embedder

        if isinstance(out_features, list):
            out_features_ = int(torch.tensor(out_features).prod())
        else:
            out_features_ = out_features

        if self.embedder is not None:
            self.embedding_layer = nn.Linear(self.embedder.get_out_channels, out_features_ * 2)
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

    def forward_w_embedding(self, x, emb=None, sample_dict={}):
        
        x = self.forward_wo_embedding(x, emb=emb, sample_dict=sample_dict)

        x_shape = x.shape

        emb_ = self.embedder(emb, sample_dict)
        scale, shift = self.embedding_layer(emb_).chunk(2, dim=-1)

        n = scale.shape[1]
        scale = scale.view(*scale.shape[:3], -1, x_shape[-1])
        shift = shift.view(*shift.shape[:3], -1, x_shape[-1])

        x = x * (scale + 1) + shift

        return x

    def forward_wo_embedding(self, x, emb=None, sample_dict=None):

        x = self.layer(x, emb=emb, sample_dict=sample_dict)

        x = x.view(*x.shape[:4],-1)

        x = self.layer_norm(x)

        return x

    def forward(self, x, emb=None, sample_dict=None):
        return self.forward_fcn(x, emb=emb, sample_dict=sample_dict)


    
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
    n_vars_total = check_get([layer_confs, kwargs, {'n_vars_total': 1}], 'n_vars_total')
    rank_channel = check_get([layer_confs, kwargs, {'rank_channel': None}], 'rank_channel')
    bias = check_get([layer_confs, kwargs, {'bias': False}], 'bias')

    if not rank_feat and not rank_vars and rank_channel is None and n_vars_total==1:
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
            n_vars_total=n_vars_total,
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

        self.in_features = int(torch.tensor(in_features).prod())
       # self.out_features = int(torch.tensor(out_features).prod())

        self.layer = nn.Linear(int(torch.tensor(in_features).prod()), int(torch.tensor(out_features).prod()), bias=bias)

        self.out_features = out_features
    def forward(self, x, **kwargs):
        
        x = x.view(*x.shape[:4], -1, self.in_features)
        x = self.layer(x)
        x = x.view(*x.shape[:4], *self.out_features)
        
        return x



class UpDownLayer(nn.Module):
   

    def __init__(self, grid_layer: GridLayer, in_features, out_features, in_zoom = None, out_zoom = None, with_nh=False, layer_confs={}):
        super().__init__()

        in_zoom = grid_layer.zoom if in_zoom is None else in_zoom

        self.grid_layer = grid_layer

        self.out_features = out_features

        if not isinstance(out_features, list):
            out_features = [out_features]

        if not isinstance(in_features, list):
            in_features = [in_features]
        
        if with_nh: 
            in_features = [self.grid_layer.adjc.shape[1]] + in_features

        if out_zoom is not None:
            zoom_diff = out_zoom - in_zoom

            if zoom_diff > 0:
                out_features = [4]*zoom_diff + out_features 

            elif zoom_diff < 0:
                in_features = [4] * abs(zoom_diff) + in_features

        self.layer = get_layer(in_features, out_features, layer_confs=layer_confs)
    

    def forward(self, x: torch.Tensor, emb= None, sample_dict: Dict = {}) -> torch.Tensor:
        
        x, mask_nh = self.grid_layer.get_nh(x, **sample_dict)
        
        x = self.layer(x, emb=emb, sample_dict=sample_dict)

        x = x.reshape(*x.shape[:3],-1,self.out_features)

        return x

