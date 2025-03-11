import torch
import torch.nn as nn
from typing import List

from ..icon_grids.grid_layer import GridLayer, MultiRelativeCoordinateManager

class NoLayer(nn.Module):

    def __init__(self,
                 rcm: MultiRelativeCoordinateManager,
                 global_level_encode: int, 
                 global_level_no: int,
                 global_level_decode: int,
                 nh_in_encode: bool=False,
                 nh_in_decode: bool=False,
                 precompute_encode: bool=True,
                 precompute_decode: bool=True) -> None: 
        
        super().__init__()
        
        rcm.register_rcm(
            global_level_encode, 
            global_level_no, 
            nh_in_encode,
            precompute_encode)

        rcm.register_rcm(
            global_level_no, 
            global_level_decode, 
            nh_in_decode,
            precompute_decode)
        
        self.rcm = rcm

        self.global_level_encode = global_level_encode
        self.global_level_decode = global_level_decode
        self.global_levels_decode_out = global_level_decode
        self.global_level_no = global_level_no
    
        self.no_nh_dist = rcm.nh_dists[global_level_no]

    def transform(self, x, coords_encode=None, coords_no=None, indices_sample=None, mask=None, emb=None):
        
        coordinates_rel, x, mask = self.rcm(self.global_level_encode, self.global_level_no, indices_sample=indices_sample, x=x, mask=mask)
            
        return self.transform_(x, coordinates_rel, mask=mask, emb=emb)
        

    def inverse_transform(self, x, coords_no=None, coords_decode=None, indices_sample=None,  mask=None, emb=None):
        
        coordinates_rel, x, mask = self.rcm(self.global_level_no, self.global_level_decode, indices_sample=indices_sample, x=x, mask=mask)

        return self.inverse_transform_(x, coordinates_rel, mask=mask, emb=emb)

    def transform_(self, x, coordinates_rel, mask=None, emb=None):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def inverse_transform_(self, x, coordinates_rel, mask=None, emb=None):
        raise NotImplementedError("This method should be implemented by subclasses.")



def add_coordinates_to_emb_dict(grid_layer: GridLayer, indices_layers, emb):

    coords = grid_layer.get_coordinates_from_grid_indices(
        indices_layers[int(grid_layer.global_level)] if indices_layers is not None else None)
    
    if emb is None:
        emb = {}

    emb['CoordinateEmbedder'] = coords

    return emb