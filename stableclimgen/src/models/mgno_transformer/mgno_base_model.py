import torch
import torch.nn as nn

from typing import List

from ...utils.helpers import get_parameter_group_from_state_dict
from ...modules.icon_grids.grid_layer import GridLayer
from ...modules.neural_operator.no_blocks import Serial_NOBlock
from .mgno_block_confs import NOBlockConfig


class MGNO_base_model(nn.Module):
    def __init__(self, 
                 mgrids,
                 block_configs: List[NOBlockConfig],
                 input_dim: int=1,
                 lifting_dim: int=1,
                 output_dim: int=1,
                 n_head_channels:int=16,
                 n_vars_total:int=1,
                 rotate_coord_system: bool=True,
                 p_dropout=0.,
                 ) -> None: 
        
                
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_vars_total = n_vars_total

        global_levels_out = [[layer_setting.get("global_level_decode", 0) 
                              for layer_setting in block_conf.layer_settings]
                              for block_conf in block_configs]
        
        global_levels_no = [[layer_setting.get("global_level_no", 0) 
                              for layer_setting in block_conf.layer_settings]
                              for block_conf in block_configs]
        
        global_levels = torch.concat((torch.tensor(global_levels_out).view(-1) 
                                     ,torch.tensor(global_levels_no).view(-1) 
                                     ,torch.tensor(0).view(-1))).unique()
        
        self.register_buffer('global_levels', global_levels, persistent=False)
        self.register_buffer('global_indices', torch.arange(mgrids[0]['coords'].shape[0]).unsqueeze(dim=0), persistent=False)
        self.register_buffer('cell_coords_global', mgrids[0]['coords'], persistent=False)
        
        # Create grid layers for each unique global level
        grid_layers = nn.ModuleDict()
        for global_level in global_levels:
            grid_layers[str(int(global_level))] = GridLayer(global_level, mgrids[global_level]['adjc_lvl'], mgrids[global_level]['adjc_mask'], mgrids[global_level]['coords'], coord_system='polar')

        self.grid_layer_0 = grid_layers["0"]
        # Construct blocks based on configurations
        self.Blocks = nn.ModuleList()

        for block_idx, block_conf in enumerate(block_configs):
            layer_settings = block_conf.layer_settings
            model_dims_out = block_conf.model_dims_out

            if block_conf.block_type == 'Serial':
                block = Serial_NOBlock(
                    lifting_dim,
                    model_dims_out,
                    grid_layers,
                    layer_settings,
                    rotate_coordinate_system=rotate_coord_system)
                
            self.Blocks.append(block)     
        
        self.out_layer = nn.Linear(block_conf.model_dims_out[-1], output_dim, bias=False)

        self.lifting_layer = nn.Linear(input_dim, lifting_dim, bias=False) if lifting_dim>1 else nn.Identity()

        

    def forward(self, x, coords_input, coords_output, indices_sample=None, mask=None, emb=None):

        """
        Forward pass for the ICON_Transformer model.

        :param x: Input tensor of shape (batch_size, num_cells, input_dim).
        :param coords_input: Input coordinates for x
        :param coords_output: Output coordinates for position embedding.
        :param sampled_indices_batch_dict: Dictionary of sampled indices for regional models.
        :param mask: Mask for dropping cells in the input tensor.
        :return: Output tensor of shape (batch_size, num_cells, output_dim).
        """

        b,n,nh,nv,nc = x.shape[:5]
        x = x.view(b,n,-1,self.input_dim)
        b,n,nv,nc = x.shape[:4]

        if mask is not None:
            mask = mask.view(*x.shape[:-1])

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

        x = self.lifting_layer(x)

        for k, block in enumerate(self.Blocks):
            
            coords_in = coords_input if k==0 else None
            coords_out = coords_output if k==len(self.Blocks)-1  else None
            
            # Process input through the block
            x, mask = block(x, coords_in=coords_in, coords_out=coords_out, indices_sample=indices_sample, mask=mask, emb=emb)

            if mask is not None:
                mask = mask.view(x.shape[:3])

        
        x = self.out_layer(x)
        x = x.view(b,n,-1)

        return x

    
    def get_global_indices_local(self, batch_sample_indices, sampled_level_fov, global_level):

        global_indices_sampled  = self.global_indices.view(-1, 4**sampled_level_fov[0])[batch_sample_indices]
        global_indices_sampled = global_indices_sampled.view(global_indices_sampled.shape[0], -1, 4**global_level)[:,:,0]   
        
        return global_indices_sampled // 4**global_level
