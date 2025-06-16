import math
import copy
from typing import List,Dict

import torch
import torch.nn as nn

from ..modules.embedding.embedder import EmbedderSequential


def get_ranks(shape, rank, rank_decay=0):
    rank_ = []
    for k in range(len(shape)):
        r = rank * (1 - rank_decay*k/(max([1,len(shape)-1]))) 
        if k < len(shape)-1:
            rank_.append(r)
        else:
            if len(rank_)>0:
                rank_.append(float(torch.tensor(rank_).mean()))
            else:
                rank_.append(float(rank))

    if rank > 1:
        ranks = [min([dim, int(rank_[k])]) for k, dim in enumerate(shape)]
    else:
        ranks = [max([1,int(dim * rank_[k])]) for k, dim in enumerate(shape)]
    
    return ranks


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
            self.layer = get_layer((), in_features, out_features, **layer_confs)


    def forward_w_embedding(self, x, emb=None, sample_dict=None):
        
        x = self.forward_wo_embedding(x, emb=emb)

        x_shape = x.shape

        emb_ = self.embedder(emb, sample_dict)
        scale, shift = self.embedding_layer(emb_).chunk(2, dim=-1)

        n = scale.shape[1]
        scale = scale.view(*scale.shape[:3], -1, x_shape[-1])
        shift = shift.view(*shift.shape[:3], -1, x_shape[-1])

        x = x * (scale + 1) + shift

        return x

    def forward_wo_embedding(self, x, emb=None, **kwargs):

        x = self.layer_norm(x)

        x = self.layer(x, emb=emb)

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
        
        self.layer1 = get_layer((1,), in_features, int(in_features * mult), layer_confs=layer_confs, bias=True)
        self.layer2 = get_layer((1,), int(in_features * mult), out_features, layer_confs=layer_confs, bias=True)
        self.dropout = nn.Dropout(p=dropout) if dropout>0 else nn.Identity()
        self.activation = nn.SiLU()

        if gamma:
            self.gamma = torch.nn.Parameter(torch.ones(out_features) * 1E-6)
            self.rtn_fcn = self.rtn_w_gamma
        else:
            self.rtn_fcn = self.rtn

    def rtn_w_gamma(self, x):
        return x + self.gamma
    
    def rtn(self, x):
        return x

    def forward(self, x, emb=None):
        
        x = self.layer1(x, emb=emb)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer2(x, emb=emb)

        return self.rtn_fcn(x)


def get_layer(
        feat_dims,
        in_features, 
        out_features,
        sum_feat_dims=True,
        layer_confs: Dict={},
        **kwargs
        ):
    
    layer_confs = copy.deepcopy(layer_confs)

    for arg in kwargs.keys():
        if arg in layer_confs:
            layer_confs.pop(arg)

    factorize = layer_confs.get('factorize', False)
    cross = layer_confs.get('cross', False)
    n_vars_total = layer_confs.get('n_vars_total', 1)

    if factorize and cross:
        layer = CrossFacLayer

    elif factorize:
        layer = FacLayer

    elif cross and n_vars_total>1:
        layer = CrossDenseLayer

    elif n_vars_total==1 and sum_feat_dims:
        layer = LinearLayer

    else:
        layer = DenseLayer
    

    layer = layer(
            feat_dims,
            in_features,
            out_features,
            sum_feat_dims=sum_feat_dims,
            **layer_confs,
            **kwargs
            )

    return layer


class LinearLayer(nn.Module):
    def __init__(self, 
                 feat_dims,
                 in_features, 
                 out_features,
                 bias=False,
                 sum_feat_dims=False,
                 **kwargs):

        super().__init__()

        self.feat_dim = int(torch.tensor(feat_dims).prod())
        in_features *= self.feat_dim
        
        if not sum_feat_dims:
            self.x_shape_out = (self.feat_dim, out_features)
            out_features *= self.feat_dim
        else:
            self.x_shape_out = (out_features,)

        self.layer = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x, emb=None):
        
        x = x.view(*x.shape[:4],-1)
        x = self.layer(x)
        x = x.view(*x.shape[:4], *self.x_shape_out)
        
        return x
    

class FacLayer(nn.Module):
    def __init__(self, 
                 feat_dims,
                 in_features, 
                 out_features, 
                 rank=0.5, 
                 const_str="btnv",
                 rank_decay=0,
                 n_vars_total=1,
                 factorize_vars=True,
                 rank_vars=4,
                 sum_feat_dims=False,
                 bias=False,
                 **kwargs):

        super().__init__()
        # Number of dimensions to transform.

        self.in_shape = (*feat_dims, in_features)

        fac_shapes = (*feat_dims, in_features, out_features)
        ranks = get_ranks(fac_shapes, rank, rank_decay=rank_decay)


        d = len(fac_shapes)

        if n_vars_total > 1 and factorize_vars:
            scale = math.sqrt(2.0 / torch.tensor((rank_vars, *ranks)).prod())
            self.core = nn.Parameter(torch.randn(rank_vars, *ranks) * scale, requires_grad=True)
            self.factors_vars = nn.Parameter(torch.ones(n_vars_total, rank_vars), requires_grad=True)
            self.contract_fun = self.contract_vars_fac

        elif n_vars_total > 1 and not factorize_vars:
            scale = math.sqrt(2.0 / torch.tensor(ranks).prod())
            self.core = nn.Parameter(torch.randn(n_vars_total, *ranks) * scale, requires_grad=True)
            self.contract_fun = self.contract_vars

        else:
            scale = math.sqrt(2.0 / torch.tensor(ranks).prod())
            self.core = nn.Parameter(torch.randn(ranks) * scale, requires_grad=True)
            self.contract_fun = self.contract

        self.factors = nn.ParameterList()
        for i, rank in enumerate(ranks):
            param = torch.empty(fac_shapes[i], rank)
            nn.init.xavier_uniform_(param)
            self.factors.append(nn.Parameter(param, requires_grad=True))

        factor_letters = "adefghijklmopqrsuwxyz" 

        core_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" 

        x_subscript = const_str + factor_letters[:(d-1)]

        core_subscript = core_letters[:d]

        factor_subscripts = [factor_letters[i] + core_letters[i] for i in range(d)]

        if n_vars_total>1 and factorize_vars:
            factor_subscripts.insert(0,'btvY')
            core_subscript = 'Y' + core_letters[:d] 

        elif n_vars_total > 1 and not factorize_vars:
            core_subscript = 'btv' +core_letters[:d] 

        else:
            core_subscript = core_letters[:d] 

        if sum_feat_dims:
            output_subscript = const_str + factor_letters[d-1]
            self.out_shape = (out_features,)
        else:
            output_subscript = const_str + factor_letters[:(d-2)] + factor_letters[d-1]
            self.out_shape = (*feat_dims, out_features)

        if bias:
            self.bias = nn.Parameter(torch.empty(*self.out_shape))
            bound_bias = 1 / math.sqrt(in_features)
            nn.init.uniform_(self.bias, -bound_bias, bound_bias)
            self.forward_fun = self.forward_with_bias
        else:
            self.bias = None
            self.forward_fun = self.forward_wo_bias
            
        self.einsum_eq = (
            f"{x_subscript},"
            f"{core_subscript},"
            f"{','.join(factor_subscripts)}"
            f"->{output_subscript}"
        )

    def contract_vars(self, x, emb=None):

        x = torch.einsum(self.einsum_eq,
                x,
                self.core[emb['VariableEmbedder']],
                *self.factors)
        return x
    
    def contract_vars_fac(self, x, emb=None):

        x = torch.einsum(
            self.einsum_eq,
            x,
            self.core,
            self.factors_vars[emb['VariableEmbedder']],
            *self.factors,
        )
        return x
    
    def contract(self, x, emb=None):

        x = torch.einsum(
            self.einsum_eq,
            x,
            self.core,
            *self.factors,
        )
        return x
    
    def forward_with_bias(self, x, emb=None):
        return self.contract_fun(x, emb=emb) + self.bias.view(*self.out_shape)
    
    def forward_wo_bias(self, x, emb=None):
        return self.contract_fun(x, emb=emb)

    def forward(self, x: torch.Tensor, emb: Dict=None):

        x = x.view(*x.shape[:4],*self.in_shape)
        x = self.forward_fun(x, emb=emb)
        x = x.reshape(*x.shape[:4],*self.out_shape)

        return x


class CrossFacLayer(nn.Module):
    def __init__(self, 
                 feat_dims, 
                 in_features, 
                 out_features, 
                 rank=0.5, 
                 const_str="btnv",
                 bias=False,
                 rank_decay=0, 
                 n_vars_total=1,
                 rank_vars=4,
                 factorize_vars=False,
                 sum_feat_dims=False,
                 **kwargs):

        super().__init__()
  
        in_shape = (*feat_dims, in_features)
        out_shape = (*feat_dims, out_features)

        in_shape = [in_sh for in_sh in in_shape if in_sh>1]
        out_shape = [out_sh for out_sh in out_shape if out_sh>1]

        d_in = len(in_shape)  
        d_out = len(out_shape)
        self.in_shape = in_shape

        ranks_in = get_ranks(in_shape, rank, no_rank_decay=rank_decay)
        ranks_out = get_ranks(out_shape, rank, no_rank_decay=rank_decay)

        fan_in = torch.tensor(ranks_in).prod()
        fan_out = torch.tensor(ranks_out).prod()

        scale = math.sqrt(2.0 / (fan_in + fan_out))
        scale_vars = 1.0 / (rank_vars)

        if n_vars_total > 1 and factorize_vars:           
            self.core = nn.Parameter(torch.randn(*ranks_out, rank_vars, *ranks_in) * scale, requires_grad=True)
            self.in_factors_vars = nn.Parameter(torch.ones(n_vars_total, rank_vars) * scale_vars, requires_grad=True)
            self.contract_fun = self.contract_vars_fac

        elif n_vars_total > 1 and not factorize_vars:
            self.core = nn.Parameter(torch.randn(n_vars_total, *ranks_out, *ranks_in) * scale, requires_grad=True)
            self.contract_fun = self.contract_vars

        else:
            self.core = nn.Parameter(torch.randn(*ranks_out, *ranks_in) * scale, requires_grad=True)
            self.contract_fun = self.contract

        out_factors = []
        for i, dim_out in enumerate(out_shape):
            param = torch.empty(dim_out, ranks_out[i])
            nn.init.xavier_uniform_(param)
            out_factors.append(nn.Parameter(param, requires_grad=True))
        
        in_factors = []
        for i, dim_in in enumerate(in_shape):
            param = torch.empty(dim_in, ranks_in[i])
            nn.init.xavier_uniform_(param)
            in_factors.append(nn.Parameter(param, requires_grad=True))


        self.in_factors = nn.ParameterList(in_factors)
        self.out_factors = nn.ParameterList(out_factors)

        in_letters = "adefghijklmß$§%"   
        out_letters = "opqrsuwxyz§%"        
        core_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ12345678" 

        in_letters_fac = in_letters
        out_letters_fac = out_letters

        if d_in > len(in_letters) or d_out > len(out_letters) or d_in+d_out > len(core_letters):
            raise ValueError("Not enough letters to label all dimensions. Increase the letter set strings.")
      
        self.x_subscript = const_str + in_letters[:d_in]
   
        out_factor_subscripts = [out_letters_fac[i] + core_letters[i] for i in range(d_out)]

        in_factor_subscripts = [in_letters_fac[i] + core_letters[d_out+ i] for i in range(d_in)]

        if n_vars_total>1 and factorize_vars:
            in_factor_subscripts.insert(0,'bvY')
            core_subscript = core_letters[:d_out] + 'Y' + core_letters[d_out:d_out+d_in]

        elif n_vars_total > 1 and not factorize_vars:
            core_subscript = 'bv' +core_letters[:d_out] + core_letters[d_out:d_out+d_in]

        else:
            core_subscript = core_letters[:d_out] + core_letters[d_out:d_out+d_in]

        if sum_feat_dims:
            output_subscript = const_str + out_letters[d_out-1]
        else:
            output_subscript = const_str + out_letters[:d_out]

        self.einsum_eq = (
            f"{self.x_subscript},"
            f"{core_subscript},"
            f"{','.join(out_factor_subscripts)},{','.join(in_factor_subscripts)}"
            f"->{output_subscript}"
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(*out_shape))
            bound_bias = 1 / math.sqrt(in_features)
            nn.init.uniform_(self.bias, -bound_bias, bound_bias)
            self.forward_fun = self.forward_with_bias
        else:
            self.bias = None
            self.forward_fun = self.forward_wo_bias

    def contract_vars(self, x, emb=None):

        x = torch.einsum(self.einsum_eq,
                x,
                self.core[emb['VariableEmbedder']],
                *self.out_factors,
                *self.in_factors)
        return x
    
    def contract_vars_fac(self, x, emb=None):

        x = torch.einsum(self.einsum_eq,
                x,
                self.core,
                *self.out_factors,
                self.in_factors_vars[emb['VariableEmbedder']],
                *self.in_factors)
        return x
    
    def contract(self, x, emb=None):

        x = torch.einsum(self.einsum_eq,
                x,
                self.core,
                *self.out_factors,
                *self.in_factors)
        return x
    
    def forward_with_bias(self, x, emb=None):
        return self.contract_fun(x, emb=emb) + self.bias.view(self.feat_dim, -1)
    
    def forward_wo_bias(self, x, emb=None):
        return self.contract_fun(x, emb=emb)

    def forward(self, x, emb=None):

        x = x.view(*x.shape[:3],*self.in_shape)

        x = self.contract_fun(x, emb=emb)
        
        x = x.reshape(*x.shape[:3],-1,x.shape[-1])
        return x
    

class DenseLayer(nn.Module):
    def __init__(self,
                 feat_dims,
                 in_features,
                 out_features,
                 rank_vars = 4,
                 n_vars_total = 1,
                 factorize_vars = False,
                 bias=False,
                 sum_feat_dims=False,
                 **kwargs
                ) -> None: 

        super().__init__()

        if sum_feat_dims:
            self.eq_out = 'btnvj'
            self.x_shape_out = (out_features) 
        else:
            self.eq_out = 'btnvmj'
            self.x_shape_out = (*feat_dims, out_features)
            
        feat_dim = int(torch.tensor(feat_dims).prod())
        self.feat_dim = feat_dim

        bound = 1 / math.sqrt(feat_dim * in_features)

        scale_vars = (1.0 / (rank_vars))

        if n_vars_total>1 and factorize_vars:
            weights = torch.empty(rank_vars, feat_dim, in_features, out_features)
            self.factors_vars = nn.Parameter(torch.ones(n_vars_total, rank_vars)*scale_vars, requires_grad=True)
            self.contract_fun = self.contract_vars_fac
        
        elif n_vars_total>1 and not factorize_vars:
            weights = torch.empty(n_vars_total, feat_dim, in_features, out_features)
            self.contract_fun = self.contract_vars

        else:
            weights = torch.empty(feat_dim, in_features, out_features)
            self.contract_fun = self.contract

        nn.init.uniform_(weights, -bound, bound)

        self.weights = nn.Parameter(weights, requires_grad=True)

        if bias:
            self.bias = nn.Parameter(torch.empty(*self.x_shape_out))
            bound_bias = 1 / math.sqrt(in_features)
            nn.init.uniform_(self.bias, -bound_bias, bound_bias)
            self.forward_fun = self.forward_with_bias
        else:
            self.bias = None
            self.forward_fun = self.forward_wo_bias

    def contract_vars_fac(self, x, emb=None):
        
        x = torch.einsum(f"btnvmi,bvf,fmij->{self.eq_out}", 
                         x, 
                         self.factors_vars[emb['VariableEmbedder']], 
                         self.weights)
        return x

    def contract_vars(self, x, emb=None):

        x = torch.einsum(f"btnvmi,bvmij->{self.eq_out}", 
                         x, 
                         self.weights[emb['VariableEmbedder']])
        return x

    def contract(self, x, emb=None):

        x = torch.einsum(f"btnvmi,mij->{self.eq_out}", 
                         x, 
                         self.weights)
        return x
    
    def forward_with_bias(self, x, emb=None):
        return self.contract_fun(x, emb=emb) + self.bias.view(self.feat_dim, -1)
    
    def forward_wo_bias(self, x, emb=None):
        return self.contract_fun(x, emb=emb)

    def forward(self, x, emb=None):

        x = x.view(*x.shape[:4],self.feat_dim,-1)

        x = self.forward_fun(x,emb=emb)

        x = x.view(*x.shape[:4],*self.x_shape_out)

        return x
    

class CrossDenseLayer(nn.Module):
    def __init__(self,
                 feat_dims,
                 in_features,
                 out_features,
                 sum_feat_dims=False,
                 rank_vars = 4,
                 n_vars_total = 1,
                 factorize_vars = False,
                 **kwargs
                ) -> None: 
         
        super().__init__()

        if sum_feat_dims:
            self.eq_out = 'btnvj'
            self.x_shape_out = (out_features) 
        else:
            self.eq_out = 'btnvmj'
            self.x_shape_out = (*feat_dims, out_features)
        
        self.x_shape_in = feat_dim * in_features

        feat_dim = int(torch.tensor(feat_dims).prod())
        self.feat_dim = feat_dim

        bound = 1 / math.sqrt(feat_dim * in_features)

        scale_vars = (1.0 / (rank_vars))

        if n_vars_total>1 and factorize_vars:
            weights = torch.empty(rank_vars, feat_dim * in_features, feat_dim, out_features)
            self.factors_vars = nn.Parameter(torch.ones(n_vars_total, rank_vars)*scale_vars, requires_grad=True)
            self.contract_fun = self.contract_vars_fac
        
        elif n_vars_total>1 and not factorize_vars:
            weights = torch.empty(n_vars_total, feat_dim * in_features, feat_dim, out_features)
            self.contract_fun = self.contract_vars

        else:
            weights = torch.empty(feat_dim * in_features, feat_dim, out_features)
            self.contract_fun = self.contract

        nn.init.uniform_(weights, -bound, bound)

        self.weights = nn.Parameter(weights, requires_grad=True)


    def contract_vars_fac(self, x, emb=None):
        
        x = torch.einsum(f"btnvmi,bvf,fimj->{self.eq_out}", 
                         x, 
                         self.factors_vars[emb['VariableEmbedder']], 
                         self.weights)
        return x

    def contract_vars(self, x, emb=None):

        x = torch.einsum(f"btnvmi,bvimj->{self.eq_out}", 
                         x, 
                         self.weights[emb['VariableEmbedder']])
        return x

    def contract(self, x, emb=None):

        x = torch.einsum(f"btnvmi,imj->{self.eq_out}", 
                         x, 
                         self.weights)
        return x

    def forward(self, x, emb=None):

        x = x.view(*x.shape[:4],self.x_shape_in)

        x = self.contract_fun(x,emb=emb)

        x = x.view(*x.shape[:4],*self.x_shape_out)

        return x