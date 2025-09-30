from typing import List,Dict
import copy

import torch.nn as nn
import torch

from .mg_base import IWD_ProjLayer,Res_UpDownLayer,ProjLayer
from ..base import IdentityLayer,get_layer,LinEmbLayer


class MG_Difference_Encoder(nn.Module):
  
    def __init__(self,
                 out_zooms: List[int]
                ) -> None: 
      
        super().__init__()

        self.out_zooms = out_zooms

        self.out_zooms = copy.deepcopy(out_zooms)

        self.out_zooms.sort(reverse = False)



    def forward(self, x_zoom: Dict[str, torch.tensor],**kwargs):
        
        in_zoom = list(x_zoom.keys())[0]
        x_in = x_zoom[in_zoom]

        x_zooms_out = {}
        b,v,t,s,c = x_in.shape
        for out_zoom in self.out_zooms:
            
            x_in = x_in.view(b, v, t, -1, 4**(in_zoom - out_zoom), c)

            x = x_in.mean(dim=-2, keepdim=True)

            x_zooms_out[out_zoom] = x.view(b,v,t,-1,c)

            x_in = (x_in - x).view(b,v,t,-1,c)

        return x_zooms_out
    
    


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
                #self.proj_layers[str(input_zoom)] = IWD_ProjLayer(grid_layers,
                #            in_zooms[k],
                #            out_zoom,
                #            interpolator_confs=interpolator_confs)
                
                self.proj_layers[str(input_zoom)] = ProjLayer(1,1,out_zoom-in_zooms[k])
            else:
                self.proj_layers[str(input_zoom)] = IdentityLayer()
        
        self.in_zooms = in_zooms


    def forward(self, x_zooms, sample_configs={}, **kwargs):

        k = 0
        for in_zoom, layer in self.proj_layers.items():
            x_out = x_zooms[int(in_zoom)]
            
            x_out = layer(x_out, sample_configs=sample_configs)

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
                

    def forward(self, x_zoom, emb=None, sample_configs={},**kwargs):

        in_zoom = list(x_zoom.keys())[0]
        x = x_zoom[in_zoom]

        for out_zoom, layer in self.layers.items():

            x_out = layer(x, emb=emb, sample_configs=sample_configs)

            x_zoom[int(out_zoom)] = x_out

        return x_zoom