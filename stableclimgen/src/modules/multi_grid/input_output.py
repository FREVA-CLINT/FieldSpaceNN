from typing import List
import copy

import torch.nn as nn

from .mg_base import IWD_ProjLayer

class MG_Difference_Encoder(nn.Module):
  
    def __init__(self,
                 in_features: List[int],
                 zoom_input: int,
                 zooms_output: List[int]
                ) -> None: 
      
        super().__init__()
        self.model_dims_out = [in_features] * len(zooms_output)
        self.output_zooms = zooms_output

        self.zooms_output = copy.deepcopy(zooms_output)
        self.zoom_input = zoom_input
        self.zooms_output.sort(reverse = True)


    def forward(self, x, coords_in=None, coords_out=None, indices_sample=None, mask_zooms=None, emb=None):
        x_zooms_out = []
        x = x[0]
        b,n,v,c = x.shape
        for idx, output_zoom in enumerate(self.zooms_output):
        
            x = x.view(b, -1, 4**(output_zoom - self.zoom_input), v, c)
            x_mean = x.mean(dim=-3)
            x = (x - x_mean.unsqueeze(dim=-3)).view(b,n,v,c)

            if mask_zooms[idx] is not None:
                mask_zooms[idx] = mask_zooms[idx].view(x.shape[:3])
            else:
                mask_zooms[idx] = None
                
            mask_zooms.append(mask_zooms[idx])
            x_zooms_out.append(x_mean)

        x_zooms_out = x_zooms_out[::-1]

        return x_zooms_out, mask_zooms
    


class MG_Sum_Decoder(nn.Module):
  
    def __init__(self,
                 grid_layers,
                 in_features: List[int],
                 out_features,
                 zooms_input: List[int],
                 zoom_output: int,
                 interpolator_confs: dict,
                 var_layer_confs: dict
                ) -> None: 
      
        super().__init__()

        self.proj_layers = nn.ModuleDict()

        self.out_features = [out_features] 
        self.output_zooms = [0]
        
        for k, input_zoom in enumerate(zooms_input):
            self.proj_layers[str(input_zoom)] = IWD_ProjLayer(grid_layers,
                          in_features[k],
                          out_features,
                          zooms_input[k],
                          zoom_output,
                          interpolator_confs=interpolator_confs,
                          var_layer_confs=var_layer_confs)
            
        self.zooms_input = zooms_input


    def forward(self, x, sample_dict=None, emb=None):

        x_zooms = dict(zip(self.zooms_input, x))

        k = 0
        for input_zoom, x in x_zooms.items():

            x_out = self.proj_layers[str(input_zoom)](x, sample_dict=sample_dict, emb=emb)

            if k == 0:
                x = x_out
            else:
                x = x + x_out

            k+=1

        return [x]