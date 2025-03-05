import torch.nn as nn

from ..icon_grids.grid_layer import RelativeCoordinateManager, GridLayer


class NoLayer(nn.Module):

    def __init__(self, 
                 grid_layer_encode: GridLayer, 
                 grid_layer_no: GridLayer,
                 grid_layer_decode: GridLayer,
                 nh_in_encode=False,
                 nh_in_decode=True,
                 precompute_encode=True,
                 precompute_decode=True,
                 rotate_coord_system=True,
                 coord_system='polar') -> None: 
        
        super().__init__()
        self.grid_layer_encode = grid_layer_encode
        self.grid_layer_decode = grid_layer_decode
        self.grid_layer_no = grid_layer_no

        self.global_level_encode = int(grid_layer_encode.global_level)
        self.global_level_decode = int(grid_layer_decode.global_level)
        self.global_level_no = int(grid_layer_no.global_level)
        self.nh_in_encode = nh_in_encode
        self.nh_in_decode = nh_in_decode
        self.rotate_coord_system = rotate_coord_system
        self.coord_system = coord_system

        self.rel_coord_mngr_encode = RelativeCoordinateManager(
            grid_layer_encode,
            grid_layer_no,
            nh_in= nh_in_encode,
            precompute = precompute_encode,
            coord_system=coord_system,
            rotate_coord_system=rotate_coord_system,
            ref='out')
        
        self.rel_coord_mngr_decode = RelativeCoordinateManager(
            grid_layer_no,
            grid_layer_decode,
            nh_in= nh_in_decode,
            precompute = precompute_decode,
            coord_system=coord_system,
            rotate_coord_system=rotate_coord_system,
            ref='out')

        self.nh_dist = grid_layer_no.nh_dist

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


            

