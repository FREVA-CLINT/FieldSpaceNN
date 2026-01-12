import math
import copy
from typing import List,Dict

from ..utils.helpers import check_get

import torch
import torch.nn as nn

from ..modules.grids.grid_layer import GridLayer
from .factorization import TuckerFacLayer,CPFacLayer


class IdentityLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, **kwargs):
        return x


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, n_groups=1, eps=1e-5, elementwise_affine=True):
        """
        normalized_shape: int or tuple, the shape over which normalization is applied
        eps: small value to avoid division by zero
        elementwise_affine: if True, learn gamma and beta
        """
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
            
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.n_groups = n_groups

        if elementwise_affine:
            if n_groups==1:
                self.weight = nn.Parameter(torch.ones(self.normalized_shape))
                self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
            else:
                self.weight = nn.Parameter(torch.ones(n_groups, *self.normalized_shape))
                self.bias = nn.Parameter(torch.zeros(n_groups, *self.normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def apply_weight_bias(self, x_hat, emb):
        if self.n_groups==1:
            return self.weight * x_hat + self.bias
        else:
            weight = self.weight[emb['GroupEmbedder']].unsqueeze(dim=2).unsqueeze(dim=2).unsqueeze(dim=2)
            bias = self.bias[emb['GroupEmbedder']].unsqueeze(dim=2).unsqueeze(dim=2).unsqueeze(dim=2)
            
            return weight * x_hat + bias

    def forward(self, x: torch.Tensor, emb=None, x_stats: torch.Tensor=None):
        if x_stats is None:
            x_stats = x

        dims = tuple(range(-len(self.normalized_shape), 0))
        mean = x_stats.mean(dim=dims, keepdim=True)
        var = x_stats.var(dim=dims, unbiased=False, keepdim=True)

        x_hat = (x - mean) / torch.sqrt(var + self.eps)

        if self.elementwise_affine:
            return self.apply_weight_bias(x_hat, emb=emb)
        else:
            return x_hat

class IdentityLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, **kwargs):
        return x
    
    def get_tensor(self, tensor, emb):
        if self.n_groups==1:
            return tensor
        else:
            return tensor[emb['GroupEmbedder']]

class EmbLayer(nn.Module):
    def __init__(self,
                 out_features,
                 embedder,
                 layer_confs_emb: dict={},
                 spatial_dim_count=1
                ) -> None: 
         
        super().__init__()

        aggregation = layer_confs_emb.get("aggregation","shift_scale")
        self.embedder = embedder
        self.spatial_dim_count = spatial_dim_count

        if not isinstance(out_features, list):
            out_features_ = [out_features]
        else:
            out_features_ = out_features
        
        self.out_features = out_features_

        in_features = [1] * (len(out_features_) - 1)   
        if aggregation == 'shift_scale':
            self.embedding_layer = get_layer([*in_features, self.embedder.get_out_channels, 1], [*out_features_, 2], layer_confs=layer_confs_emb)
            self.forward_fcn = self.forward_w_shift_scale

        elif aggregation == 'shift':
            self.embedding_layer = get_layer([*in_features, self.embedder.get_out_channels], [*out_features_], layer_confs=layer_confs_emb)
            self.forward_fcn = self.forward_w_shift
        
        elif aggregation == 'scale':
            self.embedding_layer = get_layer([*in_features, self.embedder.get_out_channels], [*out_features_], layer_confs=layer_confs_emb)
            self.forward_fcn = self.forward_w_scale

        elif aggregation == 'concat':
            self.embedding_layer = get_layer([*in_features, self.embedder.get_out_channels], [*out_features_], layer_confs=layer_confs_emb)
            self.forward_fcn = self.forward_w_concat

        self.aggregation = aggregation
    
    def forward_w_shift(self, x, emb=None, sample_configs={}):
        
        emb_ = self.embedder(emb, sample_configs)
        shift = self.embedding_layer(emb_, sample_configs=sample_configs, emb=emb)

        n = shift.shape[-1]
        shift = shift.view(*shift.shape[:3], -1, n)

        x = x + shift

        return x
    
    def forward_w_scale(self, x, emb=None, sample_configs={}):
        
        emb_ = self.embedder(emb, sample_configs)
        scale = self.embedding_layer(emb_, sample_configs=sample_configs, emb=emb)

        n = scale.shape[-1]
        scale = scale.view(*scale.shape[:3], -1, n)

        x = x * (1 + scale)

        return x

    def forward_w_concat(self, x, emb=None, sample_configs={}):
        
        emb_ = self.embedder(emb, sample_configs)
        e = self.embedding_layer(emb_, sample_configs=sample_configs, emb=emb)

        n = e.shape[-1]
        e = e.view(*e.shape[:3], -1, n)

        x = torch.concat((x, e), dim=-1)

        return x
    
    def forward_w_shift_scale(self, x, emb=None, sample_configs={}):
        
        emb_ = self.embedder(emb, sample_configs)
        scale, shift = self.embedding_layer(emb_.view(*emb_.shape[:-1],1,emb_.shape[-1]), sample_configs=sample_configs, emb=emb).chunk(2, dim=-1)

        scale = scale.squeeze(dim=-1)
        shift = shift.squeeze(dim=-1)

        x = x * (scale + 1) + shift

        return x

    def forward(self, x, emb, sample_configs={}):
        return self.forward_fcn(x, emb=emb, sample_configs=sample_configs)

class EmbIdLayer(nn.Module):
    def __init__(self,
                 out_features,
                 embedder,
                 layer_confs_emb: dict={},
                 spatial_dim_count=1
                ) -> None: 
         
        super().__init__()

        self.embedder = embedder
        self.spatial_dim_count = spatial_dim_count

        if isinstance(out_features, list):
            out_features_ = out_features[-1]
        else:
            out_features_ = out_features
        
        self.embedding_layer = get_layer([self.embedder.get_out_channels], [out_features_], layer_confs=layer_confs_emb)

    def forward(self, emb, sample_configs={}):
        
        emb_ = self.embedder(emb, sample_configs)
        shift = self.embedding_layer(emb_, sample_configs=sample_configs, emb=emb)

        shift = shift.view(*shift.shape[:3], -1, shift.shape[-1])

        return shift

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

        in_features = out_features if in_features is None else in_features

        self.embedder = embedder
        self.spatial_dim_count = spatial_dim_count

        if not isinstance(out_features, list):
            out_features_ = [out_features]
        else:
            out_features_ = out_features
        
        self.out_features = out_features
        
        if not isinstance(in_features, list):
            in_features_ = [in_features]
        else:
            in_features_ = in_features

        if self.embedder is not None:
            
            self.embedding_layer = EmbLayer(out_features, 
                                            embedder=embedder,
                                            layer_confs_emb=layer_confs_emb,
                                            spatial_dim_count=spatial_dim_count)
           
            concat = layer_confs_emb.get('aggregation','sum') == 'concat'

            self.out_features = self.embedding_layer.out_features + out_features if concat else out_features

        else:
            self.embedding_layer = IdentityLayer()

        if layer_norm:
            self.layer_norm = LayerNorm(out_features_, elementwise_affine=True, n_groups=layer_confs.get("n_groups",1))
        else:
            self.layer_norm = IdentityLayer()

        if identity_if_equal and (torch.tensor(in_features_)-torch.tensor(out_features_)==0).all():
            self.layer = IdentityLayer()
        else:
            self.layer = get_layer(in_features_, out_features_, layer_confs=layer_confs)


    def forward(self, x: torch.Tensor, emb: Dict={}, sample_configs: Dict={}, x_stats: torch.Tensor = None, **kwargs):
        
        x = self.layer(x, emb=emb, sample_configs=sample_configs)
        #x = x.reshape(*x.shape[:4 + self.spatial_dim_count-1],-1)
        x = self.layer_norm(x, emb=emb, x_stats=x_stats)

        x = self.embedding_layer(x, emb=emb, sample_configs=sample_configs)

        return x


    
class MLP_fac(nn.Module):
  
    def __init__(self,
                 in_features, 
                 out_features,
                 mult=1,
                 hidden_dim=None,
                 dropout=0,
                 layer_confs: Dict={},
                 gamma=False
                ) -> None: 
      
        super().__init__() 
        
        if hidden_dim is None:
            if isinstance(in_features, list):
                out_features_1 = [int(in_feat*mult) for in_feat in in_features]
            else:
                out_features_1 = int(in_features*mult)
        else:
            out_features_1 = hidden_dim
        

        self.layer1 = get_layer(in_features, out_features_1, layer_confs=layer_confs, bias=True)
        self.layer2 = get_layer(out_features_1, out_features, layer_confs=layer_confs, bias=True)
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
        
    ranks = check_get([layer_confs, kwargs, {'ranks': [None]}], 'ranks')
    n_groups = check_get([layer_confs, kwargs, {'n_groups': 1}], 'n_groups')
    bias = check_get([layer_confs, kwargs, {'bias': False}], 'bias')
    fac_mode = check_get([layer_confs, kwargs, {'fac_mode': 'Tucker'}], 'fac_mode')

    ranks_not_none = [rank is not None for rank in ranks]

    if not any(ranks_not_none) and n_groups==1:
        layer = LinearLayer(
                in_features,
                out_features,
                bias=bias
                )

    elif fac_mode=='Tucker':
        layer = TuckerFacLayer(
            in_features,
            out_features,
            **layer_confs)
    
    elif fac_mode == 'CP':
    
        layer = CPFacLayer(
            in_features,
            out_features,
            **layer_confs)

    return layer


class LinearLayer(nn.Module):
    def __init__(self, 
                 in_features: int|List, 
                 out_features: int|List,
                 bias=False,
                 skip_dims=None,
                 **kwargs):

        super().__init__()

        if isinstance(in_features,int):
            in_features = [in_features]

        if isinstance(out_features,int):
            out_features = [out_features]

       # self.out_features = int(torch.tensor(out_features).prod())
        self.in_features = in_features
        self.out_features = out_features
        
        self.in_shapes = in_features
        self.out_shapes = out_features

        if skip_dims is not None:
            self.in_features  = []
            self.out_features = []
            for skip_dim, in_feat, out_feat in zip(skip_dims, in_features, out_features):
                if not skip_dim:
                    self.in_features.append(in_feat)
                    self.out_features.append(out_feat)

        self.in_features_tot = math.prod(self.in_features)

        self.layer = nn.Linear(self.in_features_tot , math.prod(self.out_features), bias=bias)



    def forward(self, x, **kwargs):
        x_dims = x.shape[:5]

        x = x.reshape(*x_dims, self.in_features_tot)
        x = self.layer(x)
        x = x.view(*x_dims, *self.out_features)

        return x




