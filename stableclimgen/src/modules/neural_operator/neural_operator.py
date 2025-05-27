import torch
import torch.nn as nn
from typing import List

from ...modules.grids.grid_layer import MultiRelativeCoordinateManager

class NoLayer(nn.Module):

    def __init__(self,
                 rcm: MultiRelativeCoordinateManager,
                 in_zoom: int, 
                 no_zoom: int,
                 out_zoom: int,
                 nh_in_encode: bool=False,
                 nh_in_decode: bool=False,
                 precompute_encode: bool=True,
                 precompute_decode: bool=True,
                 coord_system: str="polar") -> None: 
        
        super().__init__()
        
        rcm.register_rcm(
            in_zoom, 
            no_zoom, 
            nh_in_encode,
            precompute_encode,
            coord_system=coord_system)

        rcm.register_rcm(
            no_zoom, 
            out_zoom, 
            nh_in_decode,
            precompute_decode,
            ref='in',
            coord_system=coord_system)
        
        self.rcm = rcm

        self.in_zoom = in_zoom
        self.out_zoom = out_zoom
        self.no_zoom = no_zoom
    
        self.no_nh_dist = rcm.nh_dists[no_zoom]

    def transform(self, x, sample_dict=None, mask=None, emb=None, **kwargs):
        
        coordinates_rel, x, mask = self.rcm(self.in_zoom, self.no_zoom, sample_dict=sample_dict, x=x, mask=mask)
            
        return self.transform_(x, coordinates_rel, mask=mask, emb=emb)
        

    def inverse_transform(self, x, sample_dict=None,  mask=None, emb=None, **kwargs):
        
        coordinates_rel, x, mask = self.rcm(self.no_zoom, self.out_zoom, sample_dict=sample_dict, x=x, mask=mask)

        return self.inverse_transform_(x, coordinates_rel, mask=mask, emb=emb)

    def transform_(self, x, coordinates_rel, mask=None, emb=None):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def inverse_transform_(self, x, coordinates_rel, mask=None, emb=None):
        raise NotImplementedError("This method should be implemented by subclasses.")