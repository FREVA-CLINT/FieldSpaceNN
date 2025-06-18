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

        if self.embedder is not None:
            self.embedding_layer = nn.Linear(self.embedder.get_out_channels, out_features * 2)
            self.forward_fcn = self.forward_w_embedding
        else:
            self.forward_fcn = self.forward_wo_embedding

        if layer_norm:
            self.layer_norm = nn.LayerNorm(out_features, elementwise_affine=True)
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

        x = self.layer_norm(x)

        x = x.view(*(x.shape[:4]), x.shape[-1])

        return x

    def forward(self, x, emb=None, sample_dict=None):
        return self.forward_fcn(x, emb=emb, sample_dict=sample_dict)


    
class MLP_fac(nn.Module):
  
    def __init__(self,
                 in_features, 
                 out_features,
                 mult=2,
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
    bias = check_get([layer_confs, kwargs, {'bias': False}], 'bias')

    if not rank_feat and not rank_vars and n_vars_total==1:
        layer = LinearLayer(
                in_features,
                out_features,
                bias=bias
                )
    else:
    
        sum_n_zooms = check_get([layer_confs, kwargs, {'sum_n_dims': 0}], 'sum_n_dims')
        base = check_get([layer_confs, kwargs, {'base': 12}], 'base')
        ranks_spatial = check_get([layer_confs, kwargs, {'ranks_spatial': {}}], 'ranks_spatial')
        dims_spatial = check_get([layer_confs, kwargs, {'dims_spatial': {}}], 'dims_spatial')
        
        layer = SpatiaFacLayer(
            in_features,
            out_features,
            ranks_spatial=ranks_spatial,
            dims_spatial=dims_spatial,
            rank_feat=rank_feat,
            rank_vars=rank_vars,
            base=base,
            n_vars_total=n_vars_total,
            sum_n_zooms=sum_n_zooms,
            bias=bias)

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

        self.in_features = in_features
        self.out_features = out_features

        self.layer = nn.Linear(int(torch.tensor(in_features).prod()), int(torch.tensor(out_features).prod()), bias=bias)

    def forward(self, x, **kwargs):
        
        x = x.view(*x.shape[:4], *self.in_features)
        x = self.layer(x)
        x = x.view(*x.shape[:4], *self.out_features)
        
        return x


"""
class SpatialSumLayer(nn.Module):
   

    def __init__(self, grid_layers: Dict[str, GridLayer], in_zoom, in_features, out_features, seq_len: int = None, out_zoom = None, with_nh=False, layer_confs={}):
        super().__init__()

        if out_zoom is not None:
            self.grid_layer_nh = grid_layers[str(out_zoom)]
            self.sum_dims = [-2,-3]

        elif seq_len is not None:
            out_zoom = in_zoom - seq_len
            self.grid_layer_nh = grid_layers[str(out_zoom)]
            self.sum_dims = [-2]

        if with_nh:
            nh_dim = self.grid_layer_nh.adjc.shape[1]
        else:
            nh_dim = 1

        in_features = [4**(out_zoom - in_zoom), nh_dim, in_features]
        
        layer = get_layer(in_features, out_features, layer_confs=layer_confs)
    
    def forward(self, x: torch.Tensor, emb: Optional[Dict] = None, mask: Optional[torch.Tensor] = None, sample_dict: Dict = None) -> torch.Tensor:
        

        x = self.proj_layer(x, emb=emb)
        
        x, x_nh = x.split([x.shape[-1]//3, 2*x.shape[-1]//3],dim=-1)

        x_nh, mask_nh = self.grid_layer.get_nh(x_nh, **sample_dict, with_nh=True, mask=mask)

        mask_nh = mask_nh.unsqueeze(dim=-1).expand(-1,-1,-1, -1, x_nh.shape[-2]) == False
        
        x = x.view(*x_nh.shape[:3],-1,*x.shape[-2:])

        b, t, s, n, v, c = x.shape

        x = rearrange(x, self.pattern)
        x_nh = rearrange(x_nh, self.pattern)
        mask_nh = rearrange(mask_nh, self.nh_mask_pattern)

        x = self.fn(x, x_nh, mask=mask_nh)

        x = rearrange(x, self.reverse_pattern, b=b, t=t, s=s, n=n, v=v, c=c)

        x = self.out_layer(x, emb=emb)

        return x

"""