import torch
import torch.nn as nn

from typing import List

from ...modules.icon_grids.grid_layer import GridLayer, MultiStepRelativeCoordinateManager, MultiRelativeCoordinateManager


class MGNO_base_model(nn.Module):
    def __init__(self, 
                 mgrids,
                 global_levels: List[int],
                 nh_up=False,
                 nh_down=True
                 ) -> None: 
        
                
        super().__init__()
        
        self.register_buffer('global_levels', global_levels, persistent=False)
        self.register_buffer('global_indices', torch.arange(mgrids[0]['coords'].shape[0]).unsqueeze(dim=0), persistent=False)
        self.register_buffer('cell_coords_global', mgrids[0]['coords'], persistent=False)
        
        # Create grid layers for each unique global level
        self.grid_layers = nn.ModuleDict()
        for global_level in global_levels:
            self.grid_layers[str(int(global_level))] = GridLayer(global_level, mgrids[global_level]['adjc_lvl'], mgrids[global_level]['adjc_mask'], mgrids[global_level]['coords'], coord_system='polar')

        self.grid_layer_0 = self.grid_layers["0"]
        # Construct blocks based on configurations

        self.rcm = MultiRelativeCoordinateManager(self.grid_layers,
                                                    coord_system="polar",
                                                    rotate_coord_system=True
                                                    )


    def prepare_coords_indices(self, coords_input=None, coords_output=None, indices_sample=None):

        if indices_sample is not None and isinstance(indices_sample, dict):
            indices_layers = dict(zip(
                self.global_levels.tolist(),
                [self.get_global_indices_local(indices_sample['sample'], 
                                               indices_sample['sample_level'], 
                                               global_level) 
                                               for global_level in self.global_levels]))
            indices_sample['indices_layers'] = indices_layers
            indices_base = indices_layers[0]
        else:           
            indices_base = indices_sample = None

        # Use global cell coordinates if none are provided
        if coords_input is None or coords_input.numel()==0:
            coords_input = self.cell_coords_global[indices_base].unsqueeze(dim=-2)

        if coords_output is None or coords_output.numel()==0:
            coords_output = self.cell_coords_global[indices_base].unsqueeze(dim=-2)
    
        return indices_sample, coords_input, coords_output
    

    def forward(self, x, coords_input=None, coords_output=None, indices_sample=None, mask=None, emb=None):
        
        indices_sample, coords_input, coords_output = self.prepare_coords_indices(coords_input,
                                                                        coords_output=coords_output, 
                                                                        indices_sample=indices_sample)
        

        return self.forward_(x, coords_input=coords_input, coords_output=coords_output, indices_sample=indices_sample, mask=mask, emb=emb)


    def forward_(x, coords_input=None, coords_output=None, indices_sample=None, mask=None, emb=None):
        pass


    def get_global_indices_local(self, batch_sample_indices, sampled_level_fov, global_level):

        global_indices_sampled  = self.global_indices.view(-1, 4**sampled_level_fov[0])[batch_sample_indices]
        global_indices_sampled = global_indices_sampled.view(global_indices_sampled.shape[0], -1, 4**global_level)[:,:,0]   
        
        return global_indices_sampled // 4**global_level
