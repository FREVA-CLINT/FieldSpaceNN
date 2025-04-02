import torch
import torch.nn as nn

from typing import List, Dict

from ...modules.icon_grids.grid_layer import GridLayer, MultiStepRelativeCoordinateManager, MultiRelativeCoordinateManager, Interpolator


class MGNO_base_model(nn.Module):
    def __init__(self, 
                 mgrids,
                 global_levels: List[int],
                 interpolate_input=False,
                 density_embedder=False,
                 interpolator_settings: Dict =None,
                 rotate_coord_system=True,
                 ) -> None:
        
                
        super().__init__()

        if interpolator_settings is not None:
            level_s = interpolator_settings.get("search_level", 3)
            if not torch.any(global_levels == level_s):
                level_s_tensor = torch.tensor([level_s])
                global_levels = torch.cat((global_levels, level_s_tensor))

            level_i = interpolator_settings.get("input_level", 0)
            if not torch.any(global_levels == level_i):
                level_i_tensor = torch.tensor([level_i])
                global_levels = torch.cat((global_levels, level_i_tensor))
        
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
                                                  rotate_coord_system=rotate_coord_system
                                                )
        
        if interpolator_settings is not None:
            self.interpolator = Interpolator(self.grid_layers,
                                             interpolator_settings.get("search_level", 3),
                                             interpolator_settings.get("input_level", 0),
                                             interpolator_settings.get("target_level", 0),
                                             interpolator_settings.get("precompute", True),
                                             interpolator_settings.get("nh_inter", 3),
                                             interpolator_settings.get("power", 1),
                                             interpolator_settings.get("new_input_level", None),
                                             interpolator_settings.get("new_search_level", None)
                                             )

        self.interpolate_input = interpolate_input
        self.density_embedder = density_embedder

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

    def prepare_batch(self, x, coords_input=None, coords_output=None, indices_sample=None, mask=None, emb=None):
        b, nt, n, nv, c = x.shape[:5]
        x = x.view(b * nt, *x.shape[2:])
        if mask is not None:
            mask = mask.view(b * nt, *mask.shape[2:])
        if coords_input is not None:
            coords_input = coords_input.view(b * nt, *coords_input.shape[2:])
        if coords_output is not None:
            coords_output = coords_output.view(b * nt, *coords_output.shape[2:])
        if indices_sample is not None and isinstance(indices_sample, dict):
            for key, value in indices_sample.items():
                indices_sample[key] = value.view(b * nt, *value.shape[2:])
        if emb is not None:
            for key, value in emb.items():
                emb[key] = value.view(b * nt, *value.shape[2:])

        return x, coords_input, coords_output, indices_sample, mask, emb
    

    def forward(self, x, coords_input=None, coords_output=None, indices_sample=None, mask=None, emb=None):
        b, nt, n = x.shape[:3]
        x, coords_input, coords_output, indices_sample, mask, emb = self.prepare_batch(x, coords_input, coords_output, indices_sample, mask, emb)
        
        indices_sample, coords_input, coords_output = self.prepare_coords_indices(coords_input,
                                                                        coords_output=coords_output, 
                                                                        indices_sample=indices_sample)
        
        if self.interpolate_input or self.density_embedder:
            interp_x, density_map = self.interpolator(x,
                                            mask=mask.unsqueeze(-1),
                                            calc_density=True,
                                            indices_sample=indices_sample)
            
            emb["DensityEmbedder"] = 1-density_map.transpose(-2,-1)
            mask =None

            if self.interpolate_input:
                x = interp_x.unsqueeze(dim=-2)

        x = self.forward_(x, coords_input=coords_input, coords_output=coords_output, indices_sample=indices_sample, mask=mask, emb=emb)
        return x.view(b, nt, *x.shape[1:]) if torch.is_tensor(x) else (x[0].view(b, nt, *x[0].shape[1:]), x[1])


    def forward_(x, coords_input=None, coords_output=None, indices_sample=None, mask=None, emb=None):
        pass


    def get_global_indices_local(self, batch_sample_indices, sampled_level_fov, global_level):
        global_indices_sampled  = self.global_indices.view(-1, 4**sampled_level_fov[0])[batch_sample_indices]
        global_indices_sampled = global_indices_sampled.view(global_indices_sampled.shape[0], -1, 4**global_level)[:,:,0]   
        
        return global_indices_sampled // 4**global_level
