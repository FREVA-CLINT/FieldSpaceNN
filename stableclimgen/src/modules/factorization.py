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
                 rank_feat = None,
                 rank_vars = None,
                 rank_channel = None,
                 n_groups = 1,
                 bias = False,
                 ranks_spatial: Dict={},
                 dims_spatial: Dict= {},
                 sum_n_zooms: int=0,
                 fc = False,
                 base=12,
                 **kwargs):

        super().__init__()

        factor_letters =  iter("adefghijklmopqruwxyz")
        core_letters = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZ") 

        self.sum_n_zooms = sum_n_zooms
        
        factorize_vars = rank_vars is not None
        factorize_features = rank_feat is not None
        factorize_channels = rank_channel is not None
        

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
        scale = 0
        if factorize_vars and n_groups>1:
            self.factor_vars = get_fac_matrix(n_groups, rank_vars)
            core_dims.append(rank_vars)
            self.get_var_fac_fcn = self.get_variable_factors
            self.get_core_fcn = self.get_core

            sub_c = next(core_letters)
            self.subscripts['core'] = sub_c
            self.subscripts['factors']['variables'] = ['bv' + sub_c]

        elif n_groups>1:
            core_dims.append(n_groups)
            #scale += n_groups
            self.get_core_fcn = self.get_core_from_var_idx
            self.get_var_fac_fcn = self.get_empty1

            self.subscripts['core'] = 'bv'

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
            self.subscripts['factors']['space'].append(sub_f + sub_c)
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

        omit_channel = (rank_channel == 0) and (in_features[-1] == out_features[-1]) 

        #if omit_channel and (len(in_features)==1 and len(in_features)==1):
        #    raise Warning here that in this case channels will not be fac
        
        if rank_channel == 0 and in_features[-1] != out_features[-1]:
            raise ValueError(
                f"When rank_channel is 0, the last dimensions of in_features ({in_features[-1]}) "
                f"and out_features ({out_features[-1]}) must be equal."
            )

        self.factors_feats_in = nn.ParameterList() 
        
        fan_in = 1
        for k, in_features_ in enumerate(in_features):
            
            if (factorize_features and k< (len(in_features) -1))  or (factorize_channels and k==(len(in_features) -1) and not omit_channel):
                rank = rank_feat if (factorize_features and k< (len(in_features) -1)) else rank_channel
                m = get_fac_matrix(in_features_, rank, init_ones=False)
                self.factors_feats_in.append(m)
                core_dims.append(m.shape[1])
                fan_in *= m.shape[1]

                sub_c = next(core_letters)
                sub_f = next(factor_letters)
                self.subscripts['core'] += sub_c
                self.subscripts['factors']['features'].append(sub_f + sub_c)
                self.subscripts['x']['features_in'] += sub_f

            elif k==(len(in_features) -1) and omit_channel:
                self.subscripts['x']['features_in'] += next(core_letters)
            else:
                sub_c = next(core_letters)
                self.subscripts['core'] += sub_c
                self.subscripts['x']['features_in'] += sub_c

                fan_in *= in_features_
                core_dims.append(in_features_)

        self.fc = fc
        if fc:
            out_features_c = [*in_features[:-1], out_features[-1]]
        else:
            out_features_c = out_features

        fan_out = 1
        self.factors_feats_out = nn.ParameterList() 
        for k, out_features_ in enumerate(out_features_c[::-1]):
            
            if (factorize_features and (fc or len(out_features)>len(in_features)) and k>0) or (factorize_channels and k==0 and not omit_channel):
                rank = rank_feat if (factorize_features and k>0) else rank_channel
                m = get_fac_matrix(out_features_, rank, init_ones=False)
                self.factors_feats_out.append(m)
                core_dims.append(m.shape[1])
                fan_out *= m.shape[1]

                sub_c = next(core_letters)
                sub_f = next(factor_letters)
                self.subscripts['core'] += sub_c
                self.subscripts['factors']['features'].append(sub_f + sub_c)

                if k <= len(out_features) - 1:
                    self.subscripts['x']['features_out'] += sub_f


            elif (not factorize_features and (fc or len(out_features)>len(in_features))) or not (factorize_channels and k==0):

                sub_c = next(core_letters)
                self.subscripts['core'] += sub_c

                core_dims.append(out_features_)
                fan_out *= out_features_

                if k <= len(out_features) - 1:
                    self.subscripts['x']['features_out'] += sub_c
                    
            elif  k==0 and omit_channel:
                self.subscripts['x']['features_out'] += self.subscripts['x']['features_in'][-1]

            elif not fc and len(out_features_c)==len(in_features):
                self.subscripts['x']['features_out'] += self.subscripts['x']['features_in'][k]


        self.feat_dims_in = in_features
        self.feat_dims_out = out_features
        
        
        core = torch.empty(core_dims)
        nn.init.normal_(core)
    #nn.init.normal_(core)
    # core = torch.ones(core_dims)# * scale
        scale = fan_in + fan_out
        self.core = nn.Parameter(core * (2 / scale)**0.5, requires_grad=True)
        
        #t = tl.tensor(core)
        self.get_in_feat_fac_fcn = self.get_in_feat_factors #if factorize_features else self.get_empty1
        self.get_out_feat_fac_fcn = self.get_out_feat_factors #if factorize_features else self.get_empty1
        self.get_space_fac_fcn = self.get_spatial_factors if factorize_space else self.get_empty_space

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
    
    def get_spatial_factors(self, sample_configs={}):
        subscripts = []

        factors = []
        if 'zoom_patch_sample' in sample_configs.keys():
            indices_zooms = global_indices_to_paths_dict(sample_configs['patch_index'], 
                                                         sizes=self.dims_spatial[:int(sample_configs['zoom_patch_sample']) + 1])
        else:
            indices_zooms = {}

        idx_f = 0
        N_out_of_sample = 0
        for zoom, factor_m in self.factors_spatial.items():
            if int(zoom) in indices_zooms.keys():
                N_out_of_sample += 1
               # subscript = 'b'
                factors.append(factor_m[indices_zooms[int(zoom)]])
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
        
        f_s, N_out_of_sample = self.get_space_fac_fcn(sample_configs=sample_configs)
        sub_space = ['b' + sub[1] if k<N_out_of_sample else sub for k, sub in enumerate(self.subscripts['factors']['space'])]

        f_v = self.get_var_fac_fcn(emb=emb)
        f_f_in = self.get_in_feat_fac_fcn()
        f_f_out = self.get_out_feat_fac_fcn()

        core = self.get_core_fcn(emb=emb)
                
        x_dims_in = [-1] + [f_s[k].shape[0] for k in range(N_out_of_sample, len(self.subscripts['x']['space']))] + self.feat_dims_in 

        x_dims_out = [-1] + self.feat_dims_out 

        x_subscripts_in = self.subscripts['x']['base'] + self.subscripts['x']['space'][N_out_of_sample:] + self.subscripts['x']['features_in']
        x_subscripts_out = self.subscripts['x']['base'] + self.subscripts['x']['space'][N_out_of_sample:] + self.subscripts['x']['features_out']

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