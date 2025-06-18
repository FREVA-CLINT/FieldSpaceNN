from typing import List,Dict
from ..modules.grids.grid_utils import global_indices_to_paths_dict

import math

import torch
import torch.nn as nn

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
    if rank < 1:
        rank = int(rank * dim)
    rank = max([rank, 1])
    
    if init_ones:
        m = torch.ones(dim, rank)/(rank)
    else:
      #  m = torch.ones(dim, rank)/(dim)
        m = torch.empty(dim, rank)
       # nn.init.kaiming_uniform_(m)
        nn.init.normal_(m)/math.sqrt(rank)
        #m = m/(dim*100)
    return nn.Parameter(m, requires_grad=True)


class SpatiaFacLayer(nn.Module):
    def __init__(self,
                 in_features: List|int, 
                 out_features: List|int, 
                 rank_feat = None,
                 rank_vars = None,
                 n_vars_total = 1,
                 bias = False,
                 ranks_spatial: Dict={},
                 dims_spatial: Dict= {},
                 sum_n_zooms: int=0,
                 base=12,
                 **kwargs):

        super().__init__()

        factor_letters =  iter("adefghijklmopqruwxyz")
        core_letters = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZ") 

        self.sum_n_zooms = sum_n_zooms
        
        factorize_vars = rank_vars is not None
        factorize_features = rank_feat is not None

        self.subscripts = {
            'factors': {
                'space': [],
                'time': [],
                'variables': [],
                'features': []
            },
            'core': '',
            'x': {
                'base': 'bvts',
                'space': '',
                'features_in':'',
                'features_out':''
            }
            }
        
        core_dims = []

        # variables -----
        scale = 1   
        if factorize_vars and n_vars_total>1:
            self.factor_vars = get_fac_matrix(n_vars_total, rank_vars)
            core_dims.append(rank_vars)
            self.get_var_fac_fcn = self.get_variable_factors
            self.get_core_fcn = self.get_core

            sub_c = next(core_letters)
            self.subscripts['core'] = sub_c
            self.subscripts['factors']['variables'] = ['bvt' + sub_c]

        elif n_vars_total>1:
            core_dims.append(n_vars_total)

            self.get_core_fcn = self.get_core_from_var_idx
            self.get_var_fac_fcn = self.get_empty1

            self.subscripts['core'] = 'bvt'

        else:
            self.get_core_fcn = self.get_core
            self.get_var_fac_fcn = self.get_empty1

        # space ---------
        
        self.min_rank = min(ranks_spatial.values()) if len(ranks_spatial)>0 else 0      
        self.factors_spatial = nn.ParameterDict()
        factorize_space = False
        
        for zoom in sorted(ranks_spatial):
            factorize_space = True
            rank = ranks_spatial[zoom]

            if len(dims_spatial)==0:
                dims = base if zoom==0 else 4
            else:
                dims = dims_spatial[zoom]
            
            self.factors_spatial[str(zoom)] = get_fac_matrix(dims, rank)
            
            sub_c = next(core_letters)
            sub_f = next(factor_letters)
            self.subscripts['core'] += sub_c
            self.subscripts['features']['space'].append(sub_f + sub_c)
            self.subscripts['x']['space']+=sub_f
            core_dims.append(rank)

        if len(dims_spatial)==0:
            dims_spatial = [4 if k>0 else 12 for k in range(10)]
        else:
            dims_spatial = list(dims_spatial.values)

        self.register_buffer('dims_spatial', torch.as_tensor(dims_spatial))

        # features ---------

        if isinstance(in_features, int):
            in_features = [in_features]

        if isinstance(out_features, int):
            out_features = [out_features]

        self.factors_feats_in = nn.ParameterList() 
        
        for in_features_ in in_features:
            
            if factorize_features:
                m = get_fac_matrix(in_features_, rank_feat, init_ones=False)
                self.factors_feats_in.append(m)
                core_dims.append(m.shape[1])
                scale /= m.shape[1]

                sub_c = next(core_letters)
                sub_f = next(factor_letters)
                self.subscripts['core'] += sub_c
                self.subscripts['factors']['features'].append(sub_f + sub_c)
                self.subscripts['x']['features_in'] += sub_f
            else:
                sub_c = next(core_letters)
                self.subscripts['core'] += sub_c
                self.subscripts['x']['features_in'] += sub_c

                scale /= in_features_
                core_dims.append(in_features_)

        self.feat_dims_in = in_features
        
        if len(in_features)==len(out_features) and len(in_features)>1:
            self.subscripts['x']['features_out'] = self.subscripts['x']['features_in'][:-1]
            self.feat_dims_out = [int(torch.tensor(in_features[:-1]).prod()), out_features[-1]]
        else:
            self.feat_dims_out = out_features
        

        if factorize_features:    
            m = get_fac_matrix(out_features[-1], rank_feat, init_ones=False)
            self.factors_feats_out = m
            core_dims.append(m.shape[1])

            sub_c = next(core_letters)
            sub_f = next(factor_letters)
            self.subscripts['core'] += sub_c
            self.subscripts['factors']['features'].append(sub_f + sub_c)
            self.subscripts['x']['features_out'] += sub_f

            scale /= m.shape[1]
        else:
            sub_c = next(core_letters)
            self.subscripts['core'] += sub_c
            self.subscripts['x']['features_out'] += sub_c
            core_dims.append(out_features[-1])
            scale /= out_features[-1]

            
        core = torch.empty(core_dims)
        #nn.init.kaiming_uniform_(core)
        nn.init.normal_(core)
        self.core = nn.Parameter(core * scale, requires_grad=True)
        
        self.get_in_feat_fac_fcn = self.get_in_feat_factors if factorize_features else self.get_empty1
        self.get_out_feat_fac_fcn = self.get_out_feat_factors if factorize_features else self.get_empty1
        self.get_space_fac_fcn = self.get_spatial_factors if factorize_space else self.get_empty_space

        if bias:
            bias = torch.empty(out_features[-1],1)
            nn.init.kaiming_uniform_(bias)
            self.bias = nn.Parameter(bias.squeeze())
            self.return_fcn = self.return_w_bias
        else:
            self.return_fcn = self.return_wo_bias

    def get_core_from_var_idx(self, emb=None):
        return self.core[emb['VariableEmbedder']]
    
    def get_core(self,**kwargs):
        return self.core
    
    def get_spatial_factors(self, sample_dict={}):
        subscripts = []

        factors = []
        if 'zoom_patch_sample' in sample_dict.keys():
            indices_zooms = global_indices_to_paths_dict(sample_dict['patch_index'], 
                                                         sizes=self.dims_spatial[:int(sample_dict['zoom_patch_sample']) + 1])
        else:
            indices_zooms = {}

        idx_f = 0
        N_out_of_sample = 0
        for zoom, factor_m in self.factors_spatial.items():
            if int(zoom) in indices_zooms.keys():
                N_out_of_sample += 1
               # subscript = 'b'
               # factors.append(factor_m[indices_zooms[int(zoom)]])
            else:
               # subscript = self.subscripts['features']['space'][idx_f]
                factors.append(factor_m)
            
            #idx_f+=1
           # subscripts.append(subscript)
        
        return factors, N_out_of_sample
    
    def get_empty_space(self,**kwargs):
        return [], 0
    
    def get_empty1(self,**kwargs):
        return []
    
    def get_variable_factors(self, emb):
        return [self.factor_vars[emb['VariableEmbedder']]]
    
    def get_in_feat_factors(self):
        return list(self.factors_feats_in) 
    
    def get_out_feat_factors(self):
        return [self.factors_feats_out]
    
    def return_w_bias(self, x):
        return x + self.bias
    
    def return_wo_bias(self, x):
        return x
    

    def forward(self, x: torch.Tensor, emb: Dict=None, sample_dict={}):
        
        f_s, N_out_of_sample = self.get_space_fac_fcn(sample_dict=sample_dict)
        sub_space = ['b' + sub[1] if k<=N_out_of_sample else sub for k, sub in enumerate(self.subscripts['factors']['space'])]

        f_v = self.get_var_fac_fcn(emb=emb)
        f_f_in = self.get_in_feat_fac_fcn()
        f_f_out = self.get_out_feat_fac_fcn()

        core = self.get_core_fcn(emb=emb)
                
        x_dims_in = [-1] + [f_s[k].shape[0] for k in range(len(self.subscripts['x']['space'][N_out_of_sample:]))] + self.feat_dims_in 

        x_dims_out = [-1] + self.feat_dims_out 

        x_subscripts_in = self.subscripts['x']['base'] + self.subscripts['x']['space'][N_out_of_sample:] + self.subscripts['x']['features_in']
        x_subscripts_out = self.subscripts['x']['base'] + self.subscripts['x']['space'][N_out_of_sample:self.sum_n_zooms] + self.subscripts['x']['features_out']

        x = x.view(*x.shape[:3], *x_dims_in)

        factor_subscripts = self.subscripts['factors']['variables'] + sub_space + self.subscripts['factors']['features'] 
        
        eq_parts_in = [x_subscripts_in] + [self.subscripts['core']] + factor_subscripts

        factors = f_v + f_s + f_f_in + f_f_out 
        
        einsum_eq = (
            f"{','.join(eq_parts_in)}"
            f"->{x_subscripts_out}"
        )

        x = torch.einsum(einsum_eq, x, core, *factors)

        x = x.reshape(*x.shape[:3], *x_dims_out)

        return self.return_fcn(x)