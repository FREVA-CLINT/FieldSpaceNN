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


def get_fac_matrix(dim, rank, init_ones=True):
    if isinstance(rank, float):
        rank = int(rank * dim)

    rank = int(max([rank, 1]))
    
    if init_ones:
        m = torch.ones(dim, rank)/(rank)
    else:
      #  m = torch.ones(dim, rank)/(dim)
        m = torch.empty(dim, rank)
       # nn.init.kaiming_uniform_(m)
        nn.init.normal_(m)
        #m = m/(dim*100)
    return nn.Parameter(m*math.sqrt(2/(rank+dim)), requires_grad=True)


class SpatiaFacLayer(nn.Module):
    def __init__(self,
                 in_features: List|int, 
                 out_features: List|int, 
                 rank_feat: int = None,
                 rank_channel: int = None,
                 skip_dims: List=None,
                 rank_vars = None,
                 n_groups = 1,
                 bias = False,
                 sum_n_zooms: int=0,
                 fc = False,
                 base=12,
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
        factor_letters =  iter("adefghijklmopqruwxyz")
        core_letters = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZ") 

        self.sum_n_zooms = sum_n_zooms
        
        factorize_vars = rank_vars is not None

        self.subscripts = {
            'factors': [],
            'core': '',
            'x_in': 'bvtn',
            'x_out': 'bvtn',
            }
        
        core_dims = []
        
        # variables -----
        scale = 0
        if factorize_vars and n_groups>1:
            self.factor_vars = get_fac_matrix(n_groups, rank_vars)
            core_dims.append(rank_vars)
            self.get_var_fac_fcn = self.get_variable_factors
            self.get_core_fcn = self.get_core

            sub_c = next(core_letters)
            self.subscripts['core'] = sub_c
            self.subscripts['factors'].append(['bv' + sub_c])

        elif n_groups>1:
            core_dims.append(n_groups)
            self.get_core_fcn = self.get_core_from_var_idx
            self.get_var_fac_fcn = self.get_empty1
            self.subscripts['core'] = 'bv'

        else:
            self.get_core_fcn = self.get_core
            self.get_var_fac_fcn = self.get_empty1

        
        ranks = [rank_feat]*(len(in_features)-1) + [rank_channel] 
        skip_dims = [False]*len(ranks) if skip_dims is None else skip_dims

        fan_in=fan_out=0
        self.factors = nn.ParameterList()
        for skip_dim, rank, f_in, f_out in zip(skip_dims, ranks, in_features, out_features):
        
            if rank == 0 and f_in != f_out:
                raise ValueError(
                    f"When rank is 0, the dimensions of in_features ({f_in}) "
                    f"and out_features ({f_out}) must be equal."
                )

            if skip_dim:
                subs = next(factor_letters)
                self.subscripts['x_in'] += subs
                self.subscripts['x_out'] += subs

            elif rank==0:
                subs = next(core_letters)
                self.subscripts['core'] += subs
                self.subscripts['x_in'] += subs
                self.subscripts['x_out'] += subs
                core_dims.append(f_in)
            
            elif rank is None:
                subs_in = next(core_letters)
                subs_out = next(core_letters)
                self.subscripts['core'] += subs_in
                self.subscripts['core'] += subs_out
                self.subscripts['x_in'] += subs_in
                self.subscripts['x_out'] += subs_out

                fan_in += f_in
                fan_out += f_out
                core_dims.append(f_in)
                core_dims.append(f_out)
            
            else:
                subs_in = next(core_letters)
                subs_out = next(core_letters)
                sub_in_f = next(factor_letters)
                sub_out_f = next(factor_letters)
                self.subscripts['core'] += subs_in
                self.subscripts['core'] += subs_out
                self.subscripts['factors'].append(sub_in_f + subs_in)
                self.subscripts['factors'].append(sub_out_f + subs_out)
                self.subscripts['x_in'] += sub_in_f
                self.subscripts['x_out'] += sub_out_f

                fan_in += rank
                fan_out += rank
                core_dims.append(rank)
                core_dims.append(rank)

                self.factors.append(get_fac_matrix(f_in, rank, init_ones=False))
                self.factors.append(get_fac_matrix(f_out, rank, init_ones=False))

        self.in_features = in_features
        self.out_features = out_features

        core = torch.empty(core_dims)
        nn.init.normal_(core)
        scale = fan_in + fan_out
        self.core = nn.Parameter(core * (2 / scale)**0.5, requires_grad=True)
        
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

    def get_core_from_var_idx(self, emb=None):
        return self.core[emb['GroupEmbedder']]
    
    def get_core(self,**kwargs):
        return self.core
    
    def get_empty1(self,**kwargs):
        return []
    
    def get_variable_factors(self, emb):
        return [self.factor_vars[emb['GroupEmbedder']]]
    
    def get_in_feat_factors(self):
        return list(self.factors_feats_in) 
    
    def get_out_feat_factors(self):
        return list(self.factors_feats_out)
    
    def return_w_bias(self, x):
        return x + self.bias
    
    def return_wo_bias(self, x):
        return x
    

    def forward(self, x: torch.Tensor, emb: Dict=None, sample_configs={}):
        
        f_v = self.get_var_fac_fcn(emb=emb)

        core = self.get_core_fcn(emb=emb)
        
        x_shape = list(x.shape[:3])
        x_dims_in = x_shape + [-1] + self.in_features 
        x_dims_out = x_shape + [-1] + self.out_features 

        x = x.view(x_dims_in)
        
        lhs = [self.subscripts['x_in'], self.subscripts['core'], *self.subscripts['factors']]
        
        einsum_eq = (
            f"{','.join(lhs)}"
            f"->{self.subscripts['x_out']}"
        )

        factors = f_v + list(self.factors)

        x = torch.einsum(einsum_eq, x, core, *factors)
        
        x = x.reshape(x_dims_out)

        return self.return_fcn(x)