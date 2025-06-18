from typing import List,Dict
import copy

import torch.nn as nn
import torch

from .mg_base import IWD_ProjLayer
from ..base import IdentityLayer,SpatiaFacLayer
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
        for in_zoom, x_in in x_zooms.items():

            x_out = self.proj_layers[str(in_zoom)](x_in, sample_dict=sample_dict)

            if k == 0:
                x = x_out
            else:
                x = x + x_out

            k+=1

        return {self.out_zooms[0]: x}