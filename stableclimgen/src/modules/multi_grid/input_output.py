from typing import List,Dict
import copy

import torch.nn as nn
import torch

from .mg_base import IWD_ProjLayer,UpDownLayer,Res_UpDownLayer
from ..base import IdentityLayer,get_layer,LinEmbLayer
class MG_Difference_Encoder(nn.Module):
  
    def __init__(self,
                 out_zooms: List[int]
                ) -> None: 
      
        super().__init__()

        self.out_zooms = out_zooms

        self.out_zooms = copy.deepcopy(out_zooms)

        self.out_zooms.sort(reverse = True)



    def forward(self, x_zoom: Dict[str, torch.tensor],**kwargs):
        
        in_zoom = list(x_zoom.keys())[0]
        x = x_zoom[in_zoom]

        b,v,t,s,c = x.shape
        for out_zoom in self.out_zooms:
        
            x = x.view(b, v, t, -1, 4**(in_zoom - out_zoom), c)
            x_coarse = x.mean(dim=-2)

            x = (x - x_coarse.unsqueeze(dim=-2)).view(b,v,t,-1,c)
          
            x_zoom[in_zoom] = x
            in_zoom = out_zoom

            x = x_coarse

        x_zoom[out_zoom] = x_coarse

        return x_zoom
    
    


class MG_Sum_Decoder(nn.Module):
  
    def __init__(self,
                 grid_layers,
                 in_zooms: List[int],
                 out_zoom: int,
                 interpolator_confs: dict={},
                ) -> None: 
      
        super().__init__()

        self.proj_layers = nn.ModuleDict()
        self.out_zooms = [out_zoom]
        
        for k, input_zoom in enumerate(in_zooms):
            if input_zoom != out_zoom:
                self.proj_layers[str(input_zoom)] = IWD_ProjLayer(grid_layers,
                            in_zooms[k],
                            out_zoom,
                            interpolator_confs=interpolator_confs)
            else:
                self.proj_layers[str(input_zoom)] = IdentityLayer()
                
        self.in_zooms = in_zooms


    def forward(self, x_zooms, sample_dict=None, **kwargs):

        k = 0
        for in_zoom, layer in self.proj_layers.items():

            x_out = layer(x_zooms[int(in_zoom)], sample_dict=sample_dict)

            if k == 0:
                x = x_out
            else:
                x = x + x_out

            k+=1

        return {self.out_zooms[0]: x}
    
class MG_Encoder(nn.Module):
  
    def __init__(self,
                 grid_layers,
                 in_zoom: int,
                 in_features,
                 out_features_list,
                 out_zooms: list,
                 layer_confs: dict={},
                ) -> None: 
      
        super().__init__()

        self.layers = nn.ModuleDict()
        self.out_zooms = out_zooms
        self.out_features = out_features_list
        
        for k, out_zoom in enumerate(out_zooms):
            if out_zoom != in_zoom:
               # self.layers[str(out_zoom)] = UpDownLayer(grid_layers[str(out_zoom)],
                #                                           in_features,
                #                                           out_features_list[k],
                #                                           out_zoom=out_zoom,
                #                                           in_zoom=in_zoom,
                #                                           with_nh=True,
                #                                           layer_confs=layer_confs)
                self.layers[str(out_zoom)] = Res_UpDownLayer(grid_layers,
                                                           in_features,
                                                           out_features_list[k],
                                                           in_zoom,
                                                           out_zoom,
                                                           with_nh=True,
                                                           layer_confs=layer_confs)
            elif in_features==out_features_list[k]:
                self.layers[str(out_zoom)] = IdentityLayer()
            else:
                self.layers[str(out_zoom)] = get_layer(in_features, out_features_list[k], layer_confs=layer_confs)
                

    def forward(self, x_zoom, emb=None, sample_dict={},**kwargs):

        in_zoom = list(x_zoom.keys())[0]
        x = x_zoom[in_zoom]

        for out_zoom, layer in self.layers.items():

            x_out = layer(x, emb=emb, sample_dict=sample_dict)

            x_zoom[int(out_zoom)] = x_out

        return x_zoom

class MG_Decoder(nn.Module):
  
    def __init__(self,
                 grid_layers,
                 in_zooms: List[int],
                 in_features_list,
                 out_features,
                 out_zoom: int,
                 aggregation = 'linear',
                 layer_confs: dict={},
                 with_residual = False
                ) -> None: 
      
        super().__init__()

        self.layers = nn.ModuleDict()
        self.out_zooms = [out_zoom]
        self.out_features = [out_features]
        
        for k, input_zoom in enumerate(in_zooms):
            if input_zoom != out_zoom:
               # self.layers[str(input_zoom)] = UpDownLayer(grid_layers[str(input_zoom)],
               #                                            in_features_list[k],
                #                                           out_features,
                #                                           out_zoom=out_zoom,
                #                                           with_nh=True,
                #                                           layer_confs=layer_confs)
                self.layers[str(input_zoom)] = Res_UpDownLayer(grid_layers,
                                                           in_features_list[k],
                                                           out_features,
                                                           input_zoom,
                                                           out_zoom,
                                                           with_nh=True,
                                                           layer_confs=layer_confs)
            else:
                self.layers[str(input_zoom)] = IdentityLayer()
        
        if aggregation == 'linear':
            layer_confs_ = copy.deepcopy(layer_confs)
            layer_confs_['rank_feat']=None
            layer_confs_['rank_channel']=None
            self.aggregation_layer = LinEmbLayer([len(self.layers), out_features], out_features,identity_if_equal=True, layer_confs=layer_confs_)

        else:
            self.aggregation_layer = IdentityLayer()

        self.aggregation = aggregation

        self.in_zooms = in_zooms
        self.with_residual = with_residual
        self.gamma = nn.Parameter(torch.ones(out_features)*1e-6, requires_grad=True)

    def forward(self, x_zooms, emb=None, sample_dict={},**kwargs):

        x_out = []
        
        for in_zoom, layer in self.layers.items():

            x = layer(x_zooms[int(in_zoom)], emb=emb, sample_dict=sample_dict)

            x_out.append(x)

        if self.aggregation == 'sum':
            x = torch.stack(x_out,dim=-1).sum(dim=-1)
        
        else:
            x = self.aggregation_layer(torch.stack(x_out, dim=-1), emb=emb, sample_dict=sample_dict)

        if self.with_residual and self.out_zooms[0] in x_zooms.keys():
            x = self.gamma*x + x_zooms[self.out_zooms[0]]

        return {self.out_zooms[0]: x}