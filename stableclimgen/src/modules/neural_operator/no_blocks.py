
from typing import List, Dict
import copy
import math

import torch
import torch.nn.functional as F
import torch.nn as nn


from .neural_operator import NoLayer

from ...modules.embedding.embedder import EmbedderSequential, get_embedder

from ..base import MLP_fac, get_layer, LinEmbLayer
from ...modules.multi_grid.mg_base import IWD_ProjLayer,ProjLayer

#tl.set_backend('pytorch')
einsum_dims = 'pqrstuvw'


class Stacked_NOBlock(nn.Module):
  
    def __init__(self,
                 Stacked_NOConv_layer,
                 with_gamma = False,
                 p_dropout=0,
                 embed_confs={},
                 layer_confs={}
                ) -> None: 
      
        super().__init__()

        self.no_conv = Stacked_NOConv_layer
        
        self.in_zooms = Stacked_NOConv_layer.in_zooms
        self.out_zooms = Stacked_NOConv_layer.out_zooms
        in_features_list = Stacked_NOConv_layer.in_features
        out_features_list = Stacked_NOConv_layer.out_features
        self.max_zoom_in = max(self.in_zooms) 

        self.lin_skip_inner = nn.ModuleDict()
        self.lin_skip_outer = nn.ModuleDict()
        self.layer_norms1 = nn.ModuleDict()
        self.layer_norms2 = nn.ModuleDict()
        self.gammas1 = nn.ParameterDict()
        self.gammas2 = nn.ParameterDict()
        self.mlp_layers = nn.ModuleDict()

        for out_zoom, out_features in out_features_list.items():

            if out_zoom in self.in_zooms:
                in_features = in_features_list[out_zoom]
                zoom_diff = 0
            else:
                zoom_diff = out_zoom - self.max_zoom_in
                in_features = in_features_list[max(in_features_list.keys())]

            embedder = get_embedder(**embed_confs, zoom=out_zoom)    

            self.lin_skip_inner[str(out_zoom)] = ProjLayer(in_features, out_features, zoom_diff)
            self.lin_skip_outer[str(out_zoom)] = ProjLayer(in_features, out_features, zoom_diff)
                
            self.layer_norms1[str(out_zoom)] = LinEmbLayer(out_features, 
                                                           out_features, 
                                                           layer_norm=True, 
                                                           identity_if_equal=True, 
                                                           layer_confs=layer_confs, 
                                                           embedder=embedder)
            
            self.layer_norms2[str(out_zoom)] = LinEmbLayer(out_features, 
                                                           out_features, 
                                                           layer_norm=True, 
                                                           identity_if_equal=True, 
                                                           layer_confs=layer_confs, 
                                                           embedder=embedder)

            if with_gamma:
                self.gammas1[str(out_zoom)] = nn.Parameter(torch.ones(out_features)*1e-6, requires_grad=True)
                self.gammas2[str(out_zoom)] = nn.Parameter(torch.ones(out_features)*1e-6, requires_grad=True)
            else:
                self.gammas1[str(out_zoom)] = nn.Parameter(torch.ones(1), requires_grad=False)
                self.gammas2[str(out_zoom)] = nn.Parameter(torch.ones(1), requires_grad=False)
            
            self.mlp_layers[str(out_zoom)] = MLP_fac(
                in_features,
                out_features,
                layer_confs={'n_groups': layer_confs['n_groups'],
                             'rank_vars':None,
                             'rank_feats':None,
                             'rank_channel':None,
                             'fc':False,
                             'bias':True})   

        self.activation = nn.SiLU()
              

    def forward(self, x_zooms, sample_configs={}, mask_zooms=None, emb=None):
        
        x_zooms_input = x_zooms

        x_zooms_out = self.no_conv(x_zooms, 
                             sample_configs=sample_configs, 
                             mask_zooms=mask_zooms, 
                             emb=emb)

        for out_zoom, x in x_zooms_out.items():

            if out_zoom in self.in_zooms:
                x_res = x_zooms_input[out_zoom]
            else:
                x_res = x_zooms_input[self.max_zoom_in]
            
            x = self.gammas1[str(out_zoom)] * self.layer_norms1[str(out_zoom)](x, emb=emb, sample_configs=sample_configs) + self.lin_skip_inner[str(out_zoom)](x_res)

            x = self.mlp_layers[str(out_zoom)](x, emb=emb)   
        
            x = self.layer_norms2[str(out_zoom)](x, emb=emb, sample_configs=sample_configs)

            x = self.gammas2[str(out_zoom)] * x + self.lin_skip_outer[str(out_zoom)](x_res)

            x = self.activation(x)

            x_zooms_out[out_zoom] = x
    
        return x_zooms_out


class Stacked_PreActivationNOBlock(nn.Module):
  
    def __init__(self,
                 Stacked_NOConv_layer,
                 with_gamma = False,
                 p_dropout=0,
                 embed_confs={},
                 layer_confs={}
                ) -> None: 
       
        super().__init__()

        self.no_conv = Stacked_NOConv_layer
        
        self.max_zoom = Stacked_NOConv_layer
        self.in_zooms = Stacked_NOConv_layer.in_zooms
        self.out_zooms = Stacked_NOConv_layer.out_zooms
        in_features_list = Stacked_NOConv_layer.in_features
        out_features_list = Stacked_NOConv_layer.out_features
        self.max_zoom_in = max(self.in_zooms) 

        self.lin_skip_inner = nn.ModuleDict()
        self.lin_skip_outer = nn.ModuleDict()
        self.layer_norms1 = nn.ModuleDict()
        self.layer_norms2 = nn.ModuleDict()
        self.gammas1 = nn.ParameterDict()
        self.gammas2 = nn.ParameterDict()
        self.mlp_layers = nn.ModuleDict()

        for in_zoom, in_features in in_features_list.items():
            embedder = get_embedder(**embed_confs, zoom=in_zoom)
            self.layer_norms1[str(in_zoom)] = LinEmbLayer(in_features, 
                                                           in_features, 
                                                           layer_norm=True, 
                                                           identity_if_equal=True, 
                                                           layer_confs=layer_confs, 
                                                           embedder=embedder)

        for out_zoom, out_features in out_features_list.items():

            if out_zoom in self.in_zooms:
                in_features = in_features_list[out_zoom]
                zoom_diff = 0
            else:
                zoom_diff = out_zoom - self.max_zoom_in
                in_features = in_features_list[max(in_features_list.keys())]

            self.lin_skip_inner[str(out_zoom)] = ProjLayer(in_features, out_features, zoom_diff)
            self.lin_skip_outer[str(out_zoom)] = ProjLayer(in_features, out_features, zoom_diff)

            embedder = get_embedder(**embed_confs, zoom=out_zoom)
            self.layer_norms2[str(out_zoom)] = LinEmbLayer(out_features, 
                                                           out_features, 
                                                           layer_norm=True, 
                                                           identity_if_equal=True, 
                                                           layer_confs=layer_confs, 
                                                           embedder=embedder)

            if with_gamma:
                self.gammas1[str(out_zoom)] = nn.Parameter(torch.ones(out_features)*1e-6, requires_grad=True)
                self.gammas2[str(out_zoom)] = nn.Parameter(torch.ones(out_features)*1e-6, requires_grad=True)
            else:
                self.gammas1[str(out_zoom)] = nn.Parameter(torch.ones(1), requires_grad=False)
                self.gammas2[str(out_zoom)] = nn.Parameter(torch.ones(1), requires_grad=False)


            self.mlp_layers[str(out_zoom)] =  MLP_fac(
                                            in_features,
                                            out_features,
                                            layer_confs={'n_groups': layer_confs['n_groups'],
                                                        'rank_vars':None,
                                                        'rank_feats':None,
                                                        'rank_channel':None,
                                                        'fc':False,
                                                        'bias':True})

        self.activation = nn.SiLU()
              

    def forward(self, x_zooms, sample_configs={}, mask_zooms=None, emb=None):
        
        x_zooms_input = x_zooms
        mask_zooms_input = mask_zooms

        x_zooms = {}
        for in_zoom in self.in_zooms:
            in_zoom = int(in_zoom)

            x_zooms[in_zoom] = self.layer_norms1[str(in_zoom)](x_zooms_input[in_zoom], emb=emb, sample_configs=sample_configs)


        x_zooms_out = self.no_conv(x_zooms, 
                             sample_configs=sample_configs, 
                             mask_zooms=mask_zooms, 
                             emb=emb)

        for out_zoom, x in x_zooms_out.items():

            if out_zoom in self.in_zooms:
                x_res = x_zooms_input[out_zoom]
            else:
                x_res = x_zooms_input[self.max_zoom_in]

            x = self.gammas1[str(out_zoom)] * x + self.lin_skip_inner[str(out_zoom)](x_res)

            x = self.layer_norms2[str(out_zoom)](x, emb=emb, sample_configs=sample_configs)

            x = self.mlp_layers[str(out_zoom)](x, emb=emb)  

            x = self.gammas2[str(out_zoom)] * x + self.lin_skip_outer[str(out_zoom)](x_res)

            x = self.activation(x)

            x_zooms_out[out_zoom] = x
    
        return x_zooms_out

class NOBlock(nn.Module):
  
    def __init__(self,
                 in_features,
                 out_features,
                 no_layer: NoLayer,
                 p_dropout=0.,
                 with_gamma = False,
                 mask_as_embedding=False,
                 OW_zero = False,
                 embed_confs={},
                 layer_confs={}
                ) -> None: 
      
        super().__init__()

        self.mask_as_embedding = mask_as_embedding

        out_zoom = int(no_layer.out_zoom)

        self.zoom_diff = (out_zoom - no_layer.in_zoom)

        zoom_diff_no_out = (no_layer.no_zoom - out_zoom)

        self.lin_skip_inner = ProjLayer(in_features, out_features, self.zoom_diff)
        self.lin_skip_outer = ProjLayer(in_features, out_features, self.zoom_diff)

        if zoom_diff_no_out == 0 and OW_zero:
            self.no_conv = O_NOConv(in_features, 
                                    out_features, 
                                    no_layer, 
                                    layer_confs=layer_confs)
        else:
            self.no_conv = NOConv(in_features, 
                                out_features, 
                                no_layer, 
                                layer_confs=layer_confs)
        
        embedder = get_embedder(**embed_confs, zoom=no_layer.out_zoom)
        self.layer_norm1 = LinEmbLayer(out_features, 
                                        out_features, 
                                        layer_norm=True, 
                                        identity_if_equal=True, 
                                        layer_confs=layer_confs, 
                                        embedder=embedder)
        
        embedder = get_embedder(**embed_confs, zoom=no_layer.out_zoom)
        self.layer_norm2 = LinEmbLayer(out_features, 
                                        out_features, 
                                        layer_norm=True, 
                                        identity_if_equal=True, 
                                        layer_confs=layer_confs, 
                                        embedder=embedder)

        if with_gamma:
            self.gamma1 = nn.Parameter(torch.ones(out_features)*1e-6, requires_grad=True)
            self.gamma2 = nn.Parameter(torch.ones(out_features)*1e-6, requires_grad=True)
        else:
            self.register_buffer('gamma1', torch.ones(out_features))
            self.register_buffer('gamma2', torch.ones(out_features))

        self.mlp_layer = MLP_fac(
            out_features,
            out_features,
            layer_confs={'n_groups': layer_confs['n_groups'],
                        'rank_vars':None,
                        'rank_feats':None,
                        'rank_channel':None,
                        'fc':False,
                        'bias':True})       

        self.activation = nn.SiLU()
              

    def forward(self, x, sample_configs={}, mask=None, emb=None):
        
        x_res = x
        
        x_conv = self.no_conv(x, 
                        sample_configs=sample_configs, 
                        mask=mask, 
                        emb=emb)

        x = self.gamma1 * self.layer_norm1(x_conv, emb=emb, sample_configs=sample_configs) + self.lin_skip_inner(x_res)

        x = self.mlp_layer(x, emb=emb)
        
        x = self.layer_norm2(x, emb=emb, sample_configs=sample_configs)

        x = self.gamma2 * x + self.lin_skip_outer(x_res)

        x = self.activation(x)
    
        return x

class PreActivation_NOBlock(nn.Module):
  
    def __init__(self,
                 in_features,
                 out_features,
                 no_layer: NoLayer,
                 p_dropout=0.,
                 with_gamma=False,
                 OW_zero = False,
                 embed_confs= {},
                 layer_confs={}
                ) -> None: 
      
        super().__init__()

        out_zoom = int(no_layer.out_zoom)

        self.zoom_diff = (out_zoom - no_layer.in_zoom)

        zoom_diff_no_out = (no_layer.no_zoom - out_zoom)

        if zoom_diff_no_out == 0 and OW_zero:
            self.no_conv = O_NOConv(in_features, 
                                    out_features, 
                                    no_layer, 
                                    layer_confs=layer_confs)
        else:
            self.no_conv = NOConv(in_features, 
                                out_features, 
                                no_layer, 
                                layer_confs=layer_confs)
        
        self.zoom_diff = (out_zoom - no_layer.in_zoom)

        embedder = get_embedder(**embed_confs, zoom=no_layer.in_zoom)

        self.layer_norm1 = LinEmbLayer(in_features, 
                                        in_features, 
                                        layer_norm=True, 
                                        identity_if_equal=True, 
                                        layer_confs=layer_confs, 
                                        embedder=embedder)
        
        embedder = get_embedder(**embed_confs, zoom=no_layer.out_zoom)
        self.layer_norm2 = LinEmbLayer(out_features, 
                                        out_features, 
                                        layer_norm=True, 
                                        identity_if_equal=True, 
                                        layer_confs=layer_confs, 
                                        embedder=embedder)

        if with_gamma:
            self.gamma1 = nn.Parameter(torch.ones(out_features)*1e-6, requires_grad=True)
            self.gamma2 = nn.Parameter(torch.ones(out_features)*1e-6, requires_grad=True)
        else:
            self.register_buffer('gamma1', torch.ones(out_features))
            self.register_buffer('gamma2', torch.ones(out_features))

        self.mlp_layer = MLP_fac(
            out_features,
            out_features,
            layer_confs={'n_groups': layer_confs['n_groups'],
                        'rank_vars':None,
                        'rank_feats':None,
                        'rank_channel':None,
                        'fc':False,
                        'bias':True})   
        
        self.activation = nn.SiLU()
        
        self.lin_skip_outer = ProjLayer(in_features, out_features, self.zoom_diff)
        self.lin_skip_inner = ProjLayer(in_features, out_features, self.zoom_diff)

    def forward(self, x, sample_configs={}, mask=None, emb=None):
        
        x_res = x
        
        x = self.layer_norm1(x, emb=emb, sample_configs=sample_configs)

        x_conv = self.no_conv(x, 
                        sample_configs=sample_configs, 
                        mask=mask, 
                        emb=emb)

        x = self.gamma1 * x_conv + self.lin_skip_inner(x_res)

        x = self.layer_norm2(x, emb=emb, sample_configs=sample_configs)

        x = self.gamma2 * self.mlp_layer(x, emb=emb) + self.lin_skip_outer(x_res)

        x = self.activation(x)
    
        return x
  

class ConcatLayer(nn.Module):
  
    def __init__(self,
                 no_dims: list,
                 in_features: int,
                 out_features: int,
                 layer_confs = {}
                ) -> None: 
      
        super().__init__()
        
        self.no_dims=no_dims
        no_dims_tot = int(torch.tensor(no_dims).prod())

        out_features = no_dims_tot*out_features

        self.layer = get_layer(in_features, out_features, layer_confs=layer_confs)

    def forward(self, x, x_c, emb=None):
       
        x_c = self.layer(x_c, emb=emb)

        x_c = x_c.view(*x.shape[:4],*self.no_dims,-1)
        x = x.view(*x.shape[:4],*self.no_dims,-1)
        x = torch.concat((x, x_c), dim=-1)

        return x

class LinearIdentityLayer(nn.Module):
    def __init__(self,
                 no_dims,
                 in_features,
                 out_features
                ) -> None: 
         
        super().__init__()

        self.no_dims = no_dims
        self.no_dims_tot = int(torch.tensor(no_dims).prod())
        
        if in_features != out_features:
            self.layer = nn.Linear(in_features, out_features)
        

    def forward(self, x, emb=None):
        
        b,n,v = x.shape[:3]
        if hasattr(self, "layer"):
        #    x = x.view(b,n,v*self.no_dims_tot,-1)
            x = x.view(b,n,v,self.no_dims_tot,-1)
            x = x.transpose(2,3)
            x = x.reshape(b,n*self.no_dims_tot,v,-1)
            x = self.layer(x, emb=emb)
            x = x.reshape(b,n,self.no_dims_tot,v,-1)
            x = x.transpose(2,3)
        x = x.reshape(b,n,v,self.no_dims_tot,-1)

        return x


class Stacked_NOConv(nn.Module):
  
    def __init__(self,
                 in_zooms: List,
                 in_features_list,
                 out_features_list,
                 no_layers: List[NoLayer],
                 out_zooms: list,
                 concat_features = 1,
                 layer_confs={}
                ) -> None: 
      
        super().__init__()

        self.register_buffer("out_zooms", torch.tensor(out_zooms))
       
        self.in_zooms = in_zooms
        self.max_zoom_in = max(in_zooms)

        self.no_layers = no_layers
        min_zoom = no_layers[-1].no_zoom
        zoom_step = no_layers[0].in_zoom - no_layers[0].no_zoom
        
        self.no_dims = {self.max_zoom_in: []}
        for k, no_layer in enumerate(no_layers):
            no_dims = copy.deepcopy(no_layer.n_params_no)
            if k>0:
                no_dims += self.no_dims[no_layer.no_zoom + zoom_step]

            self.no_dims[no_layer.no_zoom] = no_dims

        self.no_dim = copy.deepcopy(no_layer.n_params_no)

        in_features_no_max = in_features_list[0]
    
        total_concat_features = 0
        in_features_dict = dict(zip(in_zooms, in_features_list))
        self.in_features = in_features_dict

        self.zoom_concat_layers = nn.ModuleDict()
        for in_zoom in in_zooms[1:]:
            
            no_dims = self.no_dims[in_zoom]
            
            self.zoom_concat_layers[str(int(in_zoom))] = ConcatLayer(no_dims,
                    in_features_dict[in_zoom],
                    concat_features,
                    layer_confs=layer_confs)
            
            total_concat_features += concat_features

        in_features_no_max += total_concat_features
        max_features = out_features_list[-1]
        out_features_dict = dict(zip(out_zooms, out_features_list))
        self.out_features = out_features_dict

        self.zoom_reduction_layers = nn.ModuleDict()
        self.skip_gammas = nn.ParameterDict()

        for out_zoom in out_zooms:

            out_zoom = int(out_zoom)

            in_features = [*self.no_dims[out_zoom], max_features]
            out_features = [out_features_dict[out_zoom]]

            layer = get_layer(in_features, 
                            out_features,
                            layer_confs=layer_confs)

            self.zoom_reduction_layers[str(out_zoom)] = layer

        in_features = [*self.no_dims[min_zoom], in_features_no_max]
        out_features = [*self.no_dims[min_zoom], max_features]    

        self.layer = get_layer(in_features,
                               out_features,
                               layer_confs=layer_confs)


    def forward(self, x_zooms, sample_configs={}, mask_zooms=None, emb=None):
        
        x = x_zooms[self.max_zoom_in]
        mask = mask_zooms[self.max_zoom_in] if mask_zooms is not None else None

        #x_zooms = dict(zip(self.in_zooms, x_zooms))

        for k, no_layer in enumerate(self.no_layers):

            x = no_layer.transform(x, sample_configs=sample_configs, mask=mask, emb=emb)

            no_zoom = int(no_layer.no_zoom)
            if no_zoom in self.in_zooms:
                x = self.zoom_concat_layers[str(no_zoom)](x, x_zooms[no_zoom], emb=emb)

            x = x.view(*x.shape[:4],-1)

        x = self.layer(x, emb=emb)

        x_zooms_out = {}
        
        no_zoom = self.no_layers[-1].no_zoom
        if no_zoom in self.out_zooms:
            x_out = self.zoom_reduction_layers[str(no_zoom)](x, emb=emb)
            x_zooms_out[no_zoom] = x_out.reshape(*x_out.shape[:4],-1)
 
        for k, no_layer in enumerate(self.no_layers[::-1]):

            x = x.reshape(*x.shape[:4], *self.no_dim, -1)
            x = no_layer.inverse_transform(x, sample_configs=sample_configs, mask=mask, emb=emb)

            if no_layer.out_zoom in self.out_zooms:
                x_out = self.zoom_reduction_layers[str(no_layer.out_zoom)](x, emb=emb)
          
                x_zooms_out[no_layer.out_zoom] = x_out.view(*x_out.shape[:4],-1)

        return x_zooms_out

class NOConv(nn.Module):
  
    def __init__(self,
                 in_features,
                 out_features,
                 no_layer: NoLayer,
                 p_dropout=0.,
                 layer_confs={}
                ) -> None: 
      
        super().__init__()

        self.no_layer = no_layer

        self.no_dims = self.no_layer.n_params_no 
        self.in_zoom = self.no_layer.in_zoom
        self.out_zoom = int(self.no_layer.out_zoom)

        in_features = [*self.no_dims, in_features]
        out_features = [*self.no_dims, out_features]

        self.layer = get_layer(in_features, 
                                out_features, 
                                layer_confs=layer_confs)
        
    def forward(self, x, sample_configs={}, mask=None, emb=None):
        x = self.no_layer.transform(x, sample_configs=sample_configs, mask=mask, emb=emb)

        x = self.layer(x, emb=emb)
        
        x = x.view(*x.shape[:3],*self.no_dims,-1)

        x = self.no_layer.inverse_transform(x, sample_configs=sample_configs, mask=None, emb=emb)

        x = x.contiguous()

        return x

    
class O_NOConv(nn.Module):
  
    def __init__(self,
                 in_features,
                 out_features,
                 no_layer: NoLayer,
                 p_dropout=0.,
                 layer_confs={}
                ) -> None: 
      
        super().__init__()

        self.no_layer = no_layer

        self.no_dims = self.no_layer.n_params_no 
        self.in_zoom = self.no_layer.in_zoom
        self.out_zoom = int(self.no_layer.out_zoom)

        in_features = [*self.no_dims, in_features]
        out_features = [out_features]

        self.layer = get_layer(in_features, 
                                out_features, 
                                layer_confs=layer_confs)

    def forward(self, x, sample_configs={}, mask=None, emb=None):
        x = self.no_layer.transform(x, sample_configs=sample_configs, mask=mask, emb=emb)

        x = self.layer(x, emb=emb)
    
        x = x.view(*x.shape[:4],-1)

        return x