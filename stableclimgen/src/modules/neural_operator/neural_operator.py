import torch
import torch.nn as nn
from typing import List

from ..icon_grids.grid_layer import RelativeCoordinateManager, GridLayer

class NoLayer(nn.Module):

    def __init__(self, 
                 grid_layers: List[GridLayer],
                 global_level_encode: int, 
                 global_level_no: int,
                 global_level_decode: int,
                 nh_in_encode=False,
                 nh_in_decode=True,
                 precompute_encode=True,
                 precompute_decode=True,
                 rotate_coord_system=True,
                 coord_system='polar') -> None: 
        
        super().__init__()
        self.grid_layer_encode = grid_layers[str(global_level_encode)]
        self.grid_layer_decode = grid_layers[str(global_level_decode)]
        self.grid_layer_no = grid_layers[str(global_level_no)]

        self.global_level_encode = global_level_encode
        self.global_level_decode = global_level_decode
        self.global_level_no = global_level_no
        self.nh_in_encode = nh_in_encode
        self.nh_in_decode = nh_in_decode
        self.rotate_coord_system = rotate_coord_system
        self.coord_system = coord_system

        self.rel_coord_mngr_encode = RelativeCoordinateManager(
            self.grid_layer_encode,
            self.grid_layer_no,
            nh_in= nh_in_encode,
            precompute = precompute_encode,
            coord_system=coord_system,
            rotate_coord_system=rotate_coord_system,
            ref='out')
        
        self.rel_coord_mngr_decode = RelativeCoordinateManager(
            self.grid_layer_no,
            self.grid_layer_decode,
            nh_in= nh_in_decode,
            precompute = precompute_decode,
            coord_system=coord_system,
            rotate_coord_system=rotate_coord_system,
            ref='out')

        self.nh_dist = self.grid_layer_no.nh_dist

    def transform(self, x, coords_encode=None, coords_no=None, indices_sample=None, mask=None, emb=None):
        
        indices_encode  = indices_sample["indices_layers"][self.global_level_encode] if indices_sample is not None else None
        indices_no = indices_sample["indices_layers"][self.global_level_no] if indices_sample is not None else None

        coordinates_rel = self.rel_coord_mngr_encode(indices_in=indices_encode,
                                              indices_out=indices_no,
                                              coordinates_in=coords_encode,
                                              coordinates_out=coords_no,
                                              sample_dict=indices_sample)

        if self.nh_in_encode:
            x, mask = self.grid_layer_encode.get_nh(x, indices_encode, indices_sample, mask=mask)
        else:
            x = x.unsqueeze(dim=2)
            
        return self.transform_(x, coordinates_rel, mask=mask, emb=emb)
        

    def inverse_transform(self, x, coords_no=None, coords_decode=None, indices_sample=None,  mask=None, emb=None):
        
        indices_no = indices_sample["indices_layers"][self.global_level_no] if indices_sample is not None else None
        indices_decode = indices_sample["indices_layers"][self.global_level_decode] if indices_sample is not None else None

        #switch in and out, since out is the reference
        coordinates_rel = self.rel_coord_mngr_decode(indices_in=indices_no,
                                              indices_out=indices_decode,
                                              coordinates_in=coords_no,
                                              coordinates_out=coords_decode,
                                              sample_dict=indices_sample)

        self.global_level_decode
        if self.nh_in_decode:
            x, mask = self.grid_layer_no.get_nh(x, indices_no, indices_sample, mask=mask)
        else:
            x = x.unsqueeze(dim=2)
            if mask is not None:
                mask = mask.unsqueeze(dim=2)
        
        nh = x.shape[2]

        #seq_dim_in = 4**(self.global_level_no - self.global_level_decode)*nh


        return self.inverse_transform_(x, coordinates_rel, mask=mask, emb=emb)

    def transform_(self, x, coordinates_rel, mask=None, emb=None):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def inverse_transform_(self, x, coordinates_rel, mask=None, emb=None):
        raise NotImplementedError("This method should be implemented by subclasses.")


            

class StackedNoLayer(nn.Module):

    def __init__(self, 
                 grid_layers: List[GridLayer],
                 nh_up=False,
                 nh_down=True,
                 precompute=True,
                 rotate_coord_system=True,
                 coord_system='polar') -> None: 
        
        super().__init__()
        self.grid_layers = grid_layers
        self.nh_up = nh_up
        self.nh_down = nh_down
        self.rotate_coord_system = rotate_coord_system
        self.coord_system = coord_system

        self.managers_up = nn.ModuleList()
        self.managers_down = nn.ModuleList()
        
        global_levels = list(grid_layers.keys())
        
        self.register_buffer('global_levels', torch.tensor(global_levels))
        
        for idx in range(1, len(global_levels)):
            global_level_in = global_levels[idx - 1]
            global_level_out = global_levels[idx]
            
            self.managers_up.append(RelativeCoordinateManager(
                grid_layer_in=grid_layers[global_level_in],
                grid_layer_out=grid_layers[global_level_out],
                nh_in=nh_up,
                precompute=precompute,
                coord_system=coord_system,
                rotate_coord_system=rotate_coord_system
            ))
            self.managers_down.append(manager = RelativeCoordinateManager(
                grid_layer_in=grid_layers[global_level_out],
                grid_layer_out=grid_layers[global_level_in],
                nh_in=nh_down,
                precompute=precompute,
                coord_system=coord_system,
                rotate_coord_system=rotate_coord_system
            ))

    def get_manager_from_levels(self, global_level_in, global_level_out):
        index = torch.where(self.global_levels == global_level_in)

        if global_level_in-global_level_out > 0:
            return self.managers_up(index)
        else:
            return self.managers_down(index)
        
    def stacked_transform(self, x, global_level_in, global_level_out, indices_sample=None):

        for global_level_in_k in range(global_level_in, global_level_out):

            global_level_out_k = global_level_in_k + 1
            indices_in  = indices_sample["indices_layers"][global_level_in_k] if indices_sample is not None else None
            indices_out = indices_sample["indices_layers"][global_level_out_k] if indices_sample is not None else None

            coordinates_rel =self.get_manager_from_levels(global_level_in_k, global_level_out_k)(indices_in=indices_in, indices_out=indices_out, sample_dict=indices_sample)

            if self.nh_up:
                x, mask = self.grid_layers[str(global_level_encode)].get_nh(x, indices_encode, indices_sample, mask=mask)
            else:
                x = x.unsqueeze(dim=2)
            
            x, mask = self.transform_(x, coordinates_rel, mask=mask, emb=emb)
        

    def transform(self, x, coords_encode=None, coords_no=None, indices_sample=None, mask=None, emb=None):
        
        for index, global_level_encode in enumerate(self.global_levels_encode):

            global_level_no = self.global_levels_no[index]

            indices_encode  = indices_sample["indices_layers"][global_level_encode] if indices_sample is not None else None
            indices_no = indices_sample["indices_layers"][global_level_no] if indices_sample is not None else None

            coordinates_rel = self.rel_coord_mngrs_encode[index](indices_in=indices_encode,
                                                indices_out=indices_no,
                                                coordinates_in=coords_encode,
                                                coordinates_out=coords_no,
                                                sample_dict=indices_sample)

            if self.nh_in_encode:
                x, mask = self.grid_layers[str(global_level_encode)].get_nh(x, indices_encode, indices_sample, mask=mask)
            else:
                x = x.unsqueeze(dim=2)
            
            x, mask = self.transform_(x, coordinates_rel, mask=mask, emb=emb)
        
    
        emb = add_coordinates_to_emb_dict(self.grid_layers[str(global_level_no)], indices_no, emb=emb)

        return x, mask, emb
        

    def inverse_transform(self, x, coords_no=None, coords_decode=None, indices_sample=None,  mask=None, emb=None):
        
        x_out = []
        mask_out = []

        for index, global_level_decode in enumerate(self.global_levels_encode[::-1]):

            global_level_no = self.global_levels_no[index]

            indices_decode  = indices_sample["indices_layers"][global_level_decode] if indices_sample is not None else None
            indices_no = indices_sample["indices_layers"][global_level_no] if indices_sample is not None else None



            coordinates_rel = self.rel_coord_mngrs_decode[index](indices_in=indices_no,
                                                indices_out=indices_decode,
                                                coordinates_in=coords_no,
                                                coordinates_out=coords_decode,
                                                sample_dict=indices_sample)


            if self.nh_in_decode:
                x, mask = self.grid_layers[str(global_level_no)].get_nh(x, indices_no, indices_sample, mask=mask)
            else:
                x = x.unsqueeze(dim=2)
                if mask is not None:
                    mask = mask.unsqueeze(dim=2)
            
            x, mask = self.inverse_transform_(x, coordinates_rel, mask=mask, emb=emb)

            if global_level_decode in self.global_levels_decode_out:
                x_out.append(x_out)
                mask_out.append(mask_out)

        return x_out, mask_out

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