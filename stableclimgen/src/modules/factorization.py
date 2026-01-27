from typing import List,Dict
from ..modules.grids.grid_utils import global_indices_to_paths_dict

import math

import torch
import torch.nn as nn

#import tensorly as tl
#from tensorly.decomposition import tucker,parafac

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


def get_fac_matrix(dim, rank):
    if isinstance(rank, float):
        rank = int(rank * dim)

    rank = int(max([rank, 1]))
    m = torch.empty(dim, rank)
    nn.init.orthogonal_(m)

    return nn.Parameter(m, requires_grad=True)


class TuckerFacLayer(nn.Module):
    def __init__(self,
                 in_features: List|int, 
                 out_features: List|int, 
                 ranks: int = None,
                 rank_variables = None,
                 n_variables = 1,
                 bias = False,
                 **kwargs):

        super().__init__()

        # contract_dims: List[bool]
        # rank_feat: List[int]
        # ranks: None does not factorize,   

        if isinstance(in_features, int):
            in_features = [in_features]

        if isinstance(out_features, int):
            out_features = [out_features]

        assert len(in_features)==len(out_features), f"unmachting len of in_features {in_features} and out_features {out_features}"
        #contract_features = [rank_feat > 0 for k in in_features]
        self.factor_letters =  iter("aefghijklmopqruwxyz")
        self.core_letters = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZ") 

        
        factorize_vars = rank_variables is not None

        self.subscripts = {
            'factors': [],
            'core': '',
            'x_in': 'bvtnd',
            'x_out': 'bvtnd',
            }
        
        self.core_dims = []
        
        # variables -----
        scale = 0
        if factorize_vars and n_variables>1:
            self.factor_vars = get_fac_matrix(n_variables, rank_variables, init_ones=True)
            self.core_dims.append(rank_variables)
            self.get_var_fac_fcn = self.get_variable_factors
            self.get_core_fcn = self.get_core

            sub_c = next(self.core_letters)
            self.subscripts['core'] = sub_c
            self.subscripts['factors'].append('bv' + sub_c)

        elif n_variables>1:
            self.core_dims.append(n_variables)
            self.get_core_fcn = self.get_core_from_var_idx
            self.get_var_fac_fcn = self.get_empty1
            self.subscripts['core'] = 'bv'

        else:
            self.get_core_fcn = self.get_core
            self.get_var_fac_fcn = self.get_empty1

        self.factors = nn.ParameterList()
        in_dims = []
        for rank, f_in in zip(ranks, in_features):

            x_sub, core_dim = self.add(rank, f_in)
            self.subscripts['x_in'] += x_sub
            in_dims.append(core_dim)

        out_dims = []
        for rank, f_out in zip(ranks, out_features):
            
            x_sub, core_dim = self.add(rank, f_out)
            self.subscripts['x_out'] += x_sub
            out_dims.append(core_dim)

        self.in_features = in_features
        self.out_features = out_features

        core = torch.empty(self.core_dims)

        if n_variables == 1:
            core = core.reshape(math.prod(in_dims),math.prod(out_dims))
            nn.init.kaiming_normal_(core)
            core = core.reshape(self.core_dims)
        else:
            for k, c_ in enumerate(core):
                c_ = c_.reshape(math.prod(in_dims),math.prod(out_dims))
                nn.init.kaiming_normal_(c_)
                core[k] = c_.reshape(self.core_dims[1:])

        self.core = nn.Parameter(core, requires_grad=True)

        if bias:
            if len(out_features)==1:
                bias = torch.randn(out_features)
            else:
                bias = torch.empty(out_features)
                nn.init.kaiming_uniform_(bias)
                
            self.bias = nn.Parameter(bias)
            self.return_fcn = self.return_w_bias
        else:
            self.return_fcn = self.return_wo_bias
    
    def add(self, rank, features):

        core_sub = next(self.core_letters)

        if rank is not None  and rank < features and rank > 0:
            fac_sub = next(self.factor_letters)
            self.factors.append(get_fac_matrix(rank, features))
            self.subscripts['factors'] += [core_sub + fac_sub]
            core_dim = rank
            x_sub = fac_sub

        else:
            core_sub = next(self.core_letters)
            core_dim = features
            x_sub = core_sub

        self.subscripts['core'] += core_sub
        self.core_dims.append(core_dim)

        return x_sub, core_dim

    def get_core_from_var_idx(self, emb=None):
        return self.core[emb['VariableEmbedder']]
    
    def get_core(self,**kwargs):
        return self.core
    
    def get_empty1(self,**kwargs):
        return []
    
    def get_variable_factors(self, emb):
        return [self.factor_vars[emb['VariableEmbedder']]]
        
    def return_w_bias(self, x):
        return x + self.bias
    
    def return_wo_bias(self, x):
        return x
    

    def forward(self, x: torch.Tensor, emb: Dict=None, sample_configs={}):
        
        f_v = self.get_var_fac_fcn(emb=emb)

        core = self.get_core_fcn(emb=emb)
        
        x_shape = list(x.shape[:5])
        x_dims_in = x_shape + self.in_features 
        x_dims_out = x_shape + self.out_features 

        x = x.reshape(x_dims_in)
        
        lhs = [self.subscripts['x_in'], self.subscripts['core'], *self.subscripts['factors']]
        
        einsum_eq = (
            f"{','.join(lhs)}"
            f"->{self.subscripts['x_out']}"
        )

        factors = f_v + list(self.factors)

        x = torch.einsum(einsum_eq, x, core, *factors)
        
        x = x.reshape(x_dims_out)

        return self.return_fcn(x)


class CPFacLayer(nn.Module):
    def __init__(self,
                 in_features: List[int], 
                 out_features: List[int], 
                 rank: int,
                 rank_groups: int = None,
                 n_variables: int = 1,
                 keys: List[str] = [],
                 contract_feats: bool = True,
                 contract_channel: bool = True,
                 init: str = 'std_scale',
                 std: float = 0.1,
                 bias: bool = False,
                 skip_dims=None,
                 **kwargs):
        """
        CPFacLayer initializes one CP tensor per (in_features, out_features) pair.
        - Uses get_cp_tensors internally.
        - Can optionally use keys for ParameterDict.
        """

        super().__init__()
        self.n_variables = n_variables
        # Handle single int inputs
        if isinstance(in_features, int):
            in_features = [in_features]
        if isinstance(out_features, int):
            out_features = [out_features]

        assert len(in_features) == len(out_features), "`in_features` and `out_features` must be same length"
        if keys:
            assert len(keys) == len(in_features), "`keys` must match length of `in_features` and `out_features`"

        if skip_dims is not None:
            pass
        # CP tensors (using your get_cp_tensors function)
        in_features_f =[] 
        out_features_f = []
        for k,in_feats in enumerate(in_features):
            skip = skip_dims[k] if skip_dims is not None else False
            if k < len(in_features)-1 and not skip:
                in_features_f.append(in_feats)
                out_features_f.append(out_features[k])

        cp_tensors = get_cp_tensors(
            in_features=in_features_f,
            out_features=out_features_f,
            rank=rank,
            n_variables=n_variables,
            keys=keys,
            contract=contract_feats,
            init=init,
            std=std
        )

        cp_tensor = get_cp_tensor(
            in_features=in_features[-1],
            out_features=out_features[-1],
            rank=rank,
            n_variables=n_variables,
            contract=contract_channel,
            init=init,
            std=std
        )
        
        cp_tensors.append(cp_tensor)

        self.cp_tensors = cp_tensors
        self.in_feats = in_features
        self.out_feats = out_features

        # Optional: bias per output tensor
        """
        if bias:
            if keys:
                self.bias = nn.ParameterDict({
                    str(k): nn.Parameter(torch.zeros(out_f)) for k, out_f in zip(keys, out_features)
                })
            else:
                self.bias = nn.ParameterList([
                    nn.Parameter(torch.zeros(out_f)) for out_f in out_features
                ])
        else:
            self.bias = None
        """
        self.equation = get_cp_equation(len(in_features), n_variables=n_variables, contract_feats=contract_feats, contract_channel=contract_channel, skip_dims=skip_dims)

    def get_tensors(self, emb):
        return [self.get_tensor(tensor, emb) for tensor in self.cp_tensors]

    def get_tensor(self, tensor, emb):
        if self.n_variables==1:
            return tensor
        else:
            return tensor[emb['VariableEmbedder']]

    def forward(self, x, emb={}, sample_configs={}):
        x = x.view(*x.shape[:3],-1,*self.in_feats)
        x = torch.einsum(self.equation, x, *self.get_tensors(emb))

        x = x.view(*x.shape[:3],-1,*self.out_feats)
        return x

def get_cp_tensor(in_features, out_features, rank, n_variables=1, init='std_scale', std=0.1, contract=True):
    
    if contract:
        c_dims = [n_variables, in_features, out_features, rank]

    elif in_features==out_features:
        c_dims = [n_variables, out_features, rank]

    else:
        ValueError("in_features != out_features with no contraction")

    weight = torch.empty(c_dims)
    nn.init.normal_(weight)

    if init=='std_scale' and not contract:
        weight = (1 + std*weight)/math.sqrt(rank * out_features)

    elif init=='std_scale' and contract:
        weight = (1 + std*weight)/math.sqrt(rank * in_features * out_features)
        
    return nn.Parameter(weight, requires_grad=True)


def get_cp_tensors(in_features, out_features, rank, n_variables=1, keys=[], contract=True, init='std_scale',std=0.1):
    
    if len(keys)==0:
        tensors = nn.ParameterList()
        for in_f, out_f in zip(in_features,out_features):
            tensors.append(get_cp_tensor(in_f, out_f, rank, n_variables=n_variables,contract=contract, init=init, std=std))
    else:
        tensors = nn.ParameterDict()
        for key,in_f, out_f in zip(keys,in_features,out_features):
            tensors[str(key)] = get_cp_tensor(in_f, out_f, rank, n_variables=n_variables,contract=contract, init=init, std=std)
    
    return tensors


    
def get_cp_equation(n_dims, n_variables=1, contract_feats=True, contract_channel=True, nh_dim=False, skip_dims=None):

    x_letters =  iter("adefgijklmopqruwxyz")
    tensor_letters = iter("ABCDEFGHIJKLMNOPQSTUVWXYZ") 
    
    x_in_subscript = 'bvtn' if not nh_dim else 'bvtnh'
    x_out_subscript = 'bvtn' if not nh_dim else 'bvtnh'
    
    tensors_subscripts = []
    for k in range(n_dims):
        subs = '' if n_variables==1 else 'bv'
        skip = skip_dims[k] if skip_dims is not None else False

        if skip:
            sub = next(tensor_letters)
            x_in_subscript += sub
            x_out_subscript += sub

        else:
            if (contract_feats) or (contract_channel and k==n_dims-1):
                subs_in = next(tensor_letters)
                subs_out = next(tensor_letters)
                x_in_subscript += subs_in
                x_out_subscript += subs_out

                subs += subs_in + subs_out + 'R'
            else:
                subs_in_out  = next(x_letters)
                x_in_subscript += subs_in_out
                x_out_subscript += subs_in_out
                subs += subs_in_out + 'R'

        if not skip:
            tensors_subscripts.append(subs)


    lhs = f','.join([x_in_subscript] + tensors_subscripts)
    equation = f'{lhs}->{x_out_subscript}'

    return equation