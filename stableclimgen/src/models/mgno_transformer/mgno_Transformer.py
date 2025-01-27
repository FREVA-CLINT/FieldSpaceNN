import torch
import torch.nn as nn

from typing import List

from ...utils.helpers import get_parameter_group_from_state_dict
from ...modules.icon_grids.grid_layer import GridLayer
from ...modules.neural_operator.no_blocks import Serial_NOBlock,Parallel_NOBlock,UNet_NOBlock
from .mgno_block_confs import NOBlockConfig


class MGNO_Transformer(nn.Module):
    def __init__(self, 
                 mgrids,
                 block_configs: List[NOBlockConfig],
                 model_dim_in: int=1,
                 model_dim_out: int=1,
                 n_head_channels:int=16,
                 n_vars_total:int=1,
                 rotate_coord_system: bool=True,
                 pretrained_no_model_path: str=None,
                 p_dropout=0.,
                 ) -> None: 
        
                
        super().__init__()
        
        self.model_dim_in = model_dim_in
        self.model_dim_out = model_dim_out
        self.n_vars_total = n_vars_total

        global_levels_tot = [torch.tensor(block_conf.global_levels) for block_conf in block_configs] 
        global_levels = torch.concat(global_levels_tot+ [torch.tensor(0).view(-1)]).unique()
        
        self.register_buffer('global_levels', global_levels, persistent=False)
        self.register_buffer('global_indices', torch.arange(mgrids[0]['coords'].shape[0]).unsqueeze(dim=0), persistent=False)
        self.register_buffer('cell_coords_global', mgrids[0]['coords'], persistent=False)
        
        if pretrained_no_model_path is not None:
            pretrained_model_weights = torch.load(pretrained_no_model_path)['state_dict']
        else:
            pretrained_model_weights = None

        # Create grid layers for each unique global level
        grid_layers = nn.ModuleDict()
        for global_level in global_levels:
            grid_layers[str(int(global_level))] = GridLayer(global_level, mgrids[global_level]['adjc_lvl'], mgrids[global_level]['adjc_mask'], mgrids[global_level]['coords'], coord_system='polar')

        n_no_layers_total = len((torch.concat(global_levels_tot)))
        n = 0
        global_level_in = 0
        no_weights = None
        # Construct blocks based on configurations
        self.Blocks = nn.ModuleList()

        for block_idx, block_conf in enumerate(block_configs):
            
            model_d_in = model_dim_in if block_idx==0 else block.model_dim_out
            n_no_layers = len(block_conf.global_levels) 
            global_levels = block_conf.global_levels

            for k in range(n_no_layers):
                
                if pretrained_model_weights is not None:
                    no_weights = get_parameter_group_from_state_dict(pretrained_model_weights, 
                                                        f'model.Blocks.{block_idx}.NO_Blocks.{k}.no_layer',
                                                        return_reduced_keys=True)
                    
                global_level_in = 0 if block_conf.block_type != 'Stacked' and block_conf.block_type != 'UNet' or k==0 else global_level_no
                global_level_no = global_levels[k]

                layer_settings = block_conf.layer_settings[k]
                layer_settings['global_level_in'] = global_level_in
                layer_settings['global_level_out'] = global_level_no
                layer_settings['grid_layer_in'] = grid_layers[str(global_level_in)]
                layer_settings['grid_layer_no'] = grid_layers[str(global_level_no)]
                layer_settings['precompute_coordinates'] = True if n!=0 and n<n_no_layers_total else False
                layer_settings['rotate_coordinate_system'] = rotate_coord_system
                layer_settings['pretrained_weights'] = no_weights

                n+=1

            if block_conf.block_type == 'Serial':
                block = Serial_NOBlock

            elif block_conf.block_type == 'Parallel':
                block = Parallel_NOBlock

            elif block_conf.block_type == 'UNet':
                block = UNet_NOBlock

            block = block(model_d_in,
                        None if block_idx < len(block_configs)-1 else model_dim_out,
                        block_conf.layer_settings,
                        n_head_channels=n_head_channels,
                        p_dropout=p_dropout,
                        skip_mode=block_conf.skip_mode,
                        global_res=block_conf.global_res)

            self.Blocks.append(block)     
        

        

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
        x = x.view(b,n,nh,-1,1,self.model_dim_in)
        b,n,nh,nv,nc = x.shape[:5]

        if mask is not None:
            mask = mask[:,:,:,:x.shape[3]]

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


        for k, block in enumerate(self.Blocks):
            
            coords_in = coords_input if k==0 else None
            coords_out = coords_output if k==len(self.Blocks)-1  else None
            
            # Process input through the block
            x, mask = block(x, coords_in=coords_in, coords_out=coords_out, indices_sample=indices_sample, mask=mask, emb=emb)

            if mask is not None:
                mask = mask.view(x.shape[:4])

        
        x = x.view(b,n,-1)

        return x

    
    def get_global_indices_local(self, batch_sample_indices, sampled_level_fov, global_level):

        global_indices_sampled  = self.global_indices.view(-1, 4**sampled_level_fov[0])[batch_sample_indices]
        global_indices_sampled = global_indices_sampled.view(global_indices_sampled.shape[0], -1, 4**global_level)[:,:,0]   
        
        return global_indices_sampled // 4**global_level
