from typing import List,Dict

import torch
import torch.nn as nn

from ..base import get_layer, IdentityLayer
from ...modules.grids.grid_layer import GridLayer, Interpolator

class LinearReductionLayer(nn.Module):
  
    def __init__(self, 
                 in_features: List,
                 out_features: int,
                 layer_confs: Dict) -> None: 
        super().__init__()
        
        if len(in_features)>1:

            in_features =  torch.tensor(in_features)

            if ((in_features - in_features[0])>0).any():
                self.layer = get_layer(
                    (1,),
                    in_features,
                    out_features,
                    sum_feat_dims=True,
                    layer_confs=layer_confs
                )
            else:
                self.layer = get_layer(
                    (len(in_features),),
                    in_features[0],
                    out_features,
                    sum_feat_dims=True,
                    layer_confs=layer_confs
                )
        else:
            self.layer = IdentityLayer()

    def forward(self, x_levels, emb=None):

        x_out = self.layer(torch.stack(x_levels, dim=-2), emb=emb)
        x_out = x_out.view(*x_out.shape[:4],-1)

        return x_out


class SumReductionLayer(nn.Module):
  
    def __init__(self) -> None: 
        super().__init__()


    def forward(self, x_levels, mask_levels=None, emb=None):

        if mask_levels is not None and mask_levels[0] is not None:
            mask_out = torch.stack(mask_levels, dim=-1).sum(dim=-1) == len(mask_levels)
        else:
            mask_out = None
        
        x_out = torch.stack(x_levels, dim=-1).sum(dim=-1)

        return x_out, mask_out



class ProjLayer(nn.Module):
  
    def __init__(self,
                 in_features,
                 out_features,
                 zoom_diff
                ) -> None: 
      
        super().__init__() 

        self.zoom_diff = zoom_diff
        self.lin_layer = nn.Linear(in_features, out_features, bias=True) if in_features!= out_features else nn.Identity()

    def get_sum_residual(self, x, mask=None):
        if self.zoom_diff < 0:
            x = x.view(x.shape[0],x.shape[1], -1, 4**(-1*self.zoom_diff), x.shape[-2], x.shape[-1])

            if mask is not None:
                weights = mask.view(x.shape[0], x.shape[1],-1, 4**self.zoom_diff, x.shape[-2],1)==False
                weights = weights.sum(dim=-3, keepdim=True)
                x = (x/(weights+1e-10)).sum(dim=-3) 
                x = x * (weights.sum(dim=-3)!=0)

            else:
                x = x.mean(dim=-3)

        elif self.zoom_diff > 0:
            x = x.unsqueeze(dim=3).repeat_interleave(4**(self.zoom_diff), dim=3)
            x = x.view(x.shape[0],x.shape[1],-1,x.shape[-2],x.shape[-1])
            
        return x

    def forward(self, x, mask=None, **kwargs):

        return self.lin_layer(self.get_sum_residual(x, mask=mask))
    

class IWD_ProjLayer(nn.Module):
  
    def __init__(self,
                 grid_layers: List[GridLayer],
                 zoom_input: int,
                 zoom_output: int,
                 interpolator_confs: dict = {},
                ) -> None: 
      
        super().__init__()

        self.interpolators = nn.ModuleDict()
        self.lin_layers = nn.ModuleDict()
        self.output_zooms = [0]
        
        self.interpolator = Interpolator(grid_layers, 
                                        search_zoom_rel=0, 
                                        input_zoom=zoom_input, 
                                        target_zoom=zoom_output, 
                                        **interpolator_confs)


    def forward(self, x, emb=None, sample_dict=None):
      
        x,_ = self.interpolator(x.unsqueeze(dim=3), calc_density=False, sample_dict=sample_dict)

        return x