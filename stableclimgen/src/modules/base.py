import math
import copy
from typing import List,Dict

from ..utils.helpers import check_get

import torch
import torch.nn as nn

from ..modules.grids.grid_layer import GridLayer
from .factorization import SpatiaFacLayer,CPFacLayer


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
            weight = self.weight[emb['GroupEmbedder']].unsqueeze(dim=2).unsqueeze(dim=2)
            bias = self.bias[emb['GroupEmbedder']].unsqueeze(dim=2).unsqueeze(dim=2)
            
            return weight * x_hat + bias

    def forward(self, x: torch.Tensor, emb=None):

        dims = tuple(range(-len(self.normalized_shape), 0))
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, unbiased=False, keepdim=True)

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

        if isinstance(out_features, list):
            out_features_ = out_features[-1]
        else:
            out_features_ = out_features
        
        self.out_features = out_features_
                
        if aggregation == 'shift_scale':
            self.embedding_layer = get_layer([self.embedder.get_out_channels], [2, out_features_], layer_confs=layer_confs_emb)
            self.forward_fcn = self.forward_w_shift_scale

        elif aggregation == 'shift':
            self.embedding_layer = get_layer([self.embedder.get_out_channels], [out_features_], layer_confs=layer_confs_emb)
            self.forward_fcn = self.forward_w_shift
        
        elif aggregation == 'scale':
            self.embedding_layer = get_layer([self.embedder.get_out_channels], [out_features_], layer_confs=layer_confs_emb)
            self.forward_fcn = self.forward_w_scale

        elif aggregation == 'concat':
            self.embedding_layer = get_layer([self.embedder.get_out_channels], [out_features_], layer_confs=layer_confs_emb)
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
        
        x_shape = x.shape

        emb_ = self.embedder(emb, sample_configs)
        scale, shift = self.embedding_layer(emb_, sample_configs=sample_configs, emb=emb).chunk(2, dim=-2)

        n = scale.shape[1]
        scale = scale.view(*scale.shape[:3], -1, x_shape[-1])
        shift = shift.view(*shift.shape[:3], -1, x_shape[-1])

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

        if isinstance(out_features, list):
            out_features_ = out_features[-1]
        else:
            out_features_ = out_features
        
        self.out_features = out_features
        
        if isinstance(in_features, list):
            in_features_ = in_features[-1]
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

        if identity_if_equal and in_features==out_features_:
            self.layer = IdentityLayer()
        else:
            self.layer = get_layer(in_features_, out_features_, layer_confs=layer_confs)


    def forward(self, x, emb=None, sample_configs={},**kwargs):
        
        x = self.layer(x, emb=emb, sample_configs=sample_configs)
        x = x.view(*x.shape[:4 + self.spatial_dim_count-1],-1)
        x = self.layer_norm(x, emb=emb)

        x = self.embedding_layer(x, emb=emb, sample_configs=sample_configs)

        return x


    
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
    fac_mode = check_get([layer_confs, kwargs, {'fac_mode': 'Tucker'}], 'fac_mode')
    skip_dims = check_get([layer_confs, kwargs, {'skip_dims': None}], 'skip_dims')

    if not rank_feat and not rank_vars and rank_channel is None and n_groups==1:
        layer = LinearLayer(
                in_features,
                out_features,
                bias=bias,
                skip_dims = skip_dims
                )

    elif fac_mode=='Tucker':
        layer = SpatiaFacLayer(
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

        self.in_features = [int(torch.tensor(self.in_features).prod())]
        self.out_features = [int(torch.tensor(self.out_features).prod())]

        self.layer = nn.Linear(self.in_features[0], self.out_features[0], bias=bias)



    def forward(self, x, **kwargs):
        x_dims = x.shape[:-(len(self.in_features) + 1)]

        x = x.reshape(*x_dims, -1, *self.in_features)
        x = self.layer(x)
        x = x.view(*x.shape[:3], -1, x.shape[-1])

        return x




