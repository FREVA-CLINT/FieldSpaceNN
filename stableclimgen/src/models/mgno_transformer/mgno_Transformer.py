import torch
import torch.nn as nn

import copy
from typing import List, Dict
import omegaconf

from ...utils.helpers import get_parameter_group_from_state_dict
from ...modules.icon_grids.grid_layer import GridLayer
from ...modules.neural_operator.no_blocks import Serial_NOBlock,Stacked_NOBlock,Parallel_NOBlock,UNet_NOBlock

from ...modules.neural_operator.neural_operator import get_no_layer



def check_value(value, n_repeat):
    if not isinstance(value, list) and not isinstance(value, omegaconf.listconfig.ListConfig):
        value = [value]*n_repeat
    elif (isinstance(value, list) or isinstance(value, omegaconf.listconfig.ListConfig)) and len(value)<=1 and len(value)< n_repeat:
        value = [list(value) for _ in range(n_repeat)] if len(value)==0 else list(value)*n_repeat
    return value


class NOBlockConfig:

    def __init__(self, 
                 block_type: str,
                 neural_operator_type: int|list,
                 global_levels: int|list, 
                 model_dims_out: int|list,
                 n_params: List[int],
                 att_block_types_encode: List[bool],
                 global_params_learnable : List[bool],
                 att_block_types_decode: List[bool] = [],
                 global_params_init : List[float] = [],
                 embed_names_encode: List[List[str]] = None,
                 embed_names_decode: List[List[str]] = None,
                 embed_confs: Dict = None,
                 embed_mode: str = "sum",
                 nh_transformation: bool = False,
                 nh_inverse_transformation: bool = False,
                 att_dims: int = None,
                 random_amplitudes: bool = False,
                 n_var_amplitudes: int = 1,
                 multi_grid_attention: bool=False,
                 neural_operator_type_nh: str='polNormal',
                 n_params_nh: List[int] = [[4,2]],
                 global_params_init_nh: List[float] = [[3.0]],
                 spatial_attention_configs: dict = {},
                 global_res=False):

        n_no_layers = len(global_levels)

        inputs = copy.deepcopy(locals())
        self.block_type = block_type
        self.multi_grid_attention = multi_grid_attention
        self.global_res = global_res

        for input, value in inputs.items():
            if input != 'self' and input != 'block_type' and input !=multi_grid_attention and input != "global_res":
                setattr(self, input, check_value(value, n_no_layers))
        

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

            n_no_layers = len(block_conf.global_levels) 
            global_levels = block_conf.global_levels

            no_layers = []
            nh_no_layers = []
            for k in range(n_no_layers):
                
                global_level_in = 0 if block_conf.block_type != 'Stacked' and block_conf.block_type != 'UNet' or k==0 else global_level_no
                global_level_no = global_levels[k]

                if pretrained_model_weights is not None:
                    no_weights = get_parameter_group_from_state_dict(pretrained_model_weights, 
                                                        f'model.Blocks.{block_idx}.NO_Blocks.{k}.no_layer',
                                                        return_reduced_keys=True)

                no_layer = get_no_layer(block_conf.neural_operator_type[k], 
                                    grid_layers, 
                                    global_level_in, 
                                    global_level_no,
                                    n_params=block_conf.n_params[k],
                                    params_init=block_conf.global_params_init[k],
                                    params_learnable=block_conf.global_params_learnable[k],
                                    nh_projection=block_conf.nh_transformation[k],
                                    nh_backprojection=block_conf.nh_inverse_transformation[k],
                                    precompute_coordinates=True if n!=0 and n<n_no_layers_total else False,
                                    rotate_coordinate_system=rotate_coord_system,
                                    pretrained_weights=no_weights,
                                    random_amplitudes=block_conf.random_amplitudes[k],
                                    n_var_amplitudes=block_conf.n_var_amplitudes[k])

                no_layers.append(no_layer)

                nh_no_layer_required_encode = torch.tensor(['nh' in block_conf.att_block_types_encode[k]]).any()
                nh_no_layer_required_decode = torch.tensor(['nh' in block_conf.att_block_types_decode[k]]).any()
                nh_no_layer_required = nh_no_layer_required_encode or nh_no_layer_required_decode

                if nh_no_layer_required:
                    
                    no_layer_nh = get_no_layer(block_conf.neural_operator_type_nh[k], 
                                    grid_layers, 
                                    global_level_no, 
                                    global_level_no,
                                    n_params=block_conf.n_params_nh[k],
                                    params_init=block_conf.global_params_init_nh[k],
                                    params_learnable=block_conf.global_params_learnable[k],
                                    nh_projection=True,
                                    nh_backprojection=True,
                                    precompute_coordinates=True,
                                    rotate_coordinate_system=rotate_coord_system)
                else:
                    no_layer_nh = None
                
                nh_no_layers.append(no_layer_nh)

                n+=1

            if block_conf.block_type == 'Serial':
                block = Serial_NOBlock

            elif block_conf.block_type == 'Stacked':
                block = Stacked_NOBlock

            elif block_conf.block_type == 'Parallel':
                block = Parallel_NOBlock

            elif block_conf.block_type == 'UNet':
                block = UNet_NOBlock


            self.Blocks.append(block(model_dim_in,
                        block_conf.model_dims_out,
                        no_layers,
                        n_params=block_conf.n_params,
                        att_block_types_encode=block_conf.att_block_types_encode,
                        att_block_types_decode=block_conf.att_block_types_decode,
                        embed_names_encode=block_conf.embed_names_encode,
                        embed_names_decode=block_conf.embed_names_decode,
                        embed_confs=block_conf.embed_confs,
                        embed_mode=block_conf.embed_mode,
                        n_head_channels=n_head_channels,
                        att_dims=block_conf.att_dims,
                        no_layers_nh=nh_no_layers,
                        multi_grid_attention=block_conf.multi_grid_attention,
                        spatial_attention_configs=block_conf.spatial_attention_configs,
                        p_dropout=p_dropout,
                        global_res=block_conf.global_res))     
        
        

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
        x = x.view(b,n,nh,-1,self.model_dim_in)
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
            x = x.view(b,n,-1,nv,nc)
            if mask is not None:
                mask = mask.view(x.shape[:4])

        
        x = x.view(b,n,-1)

        return x

    
    def get_global_indices_local(self, batch_sample_indices, sampled_level_fov, global_level):

        global_indices_sampled  = self.global_indices.view(-1, 4**sampled_level_fov[0])[batch_sample_indices]
        global_indices_sampled = global_indices_sampled.view(global_indices_sampled.shape[0], -1, 4**global_level)[:,:,0]   
        
        return global_indices_sampled // 4**global_level
