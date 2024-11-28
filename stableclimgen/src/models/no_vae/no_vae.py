import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
from typing import List
import xarray as xr
import omegaconf

from ...utils.grid_utils_icon import icon_grid_to_mgrid

from ...modules.icon_grids.icon_grids import GridLayer
from ...modules.neural_operator.no_blocks import Stacked_NOBlock, Serial_NOBlock, Parallel_NOBlock
from ..vae.quantization import Quantization

from ...modules.neural_operator.neural_operator import Normal_VM_NoLayer, Normal_NoLayer, FT_NOLayer



def check_value(value, n_repeat):
    if not isinstance(value, list) and not isinstance(value, omegaconf.listconfig.ListConfig):
        value = [value]*n_repeat
    elif (isinstance(value, list) or isinstance(value, omegaconf.listconfig.ListConfig)) and len(value)<=1 and len(value)< n_repeat:
        value = [list(value) for _ in range(n_repeat)] if len(value)==0 else list(value)*n_repeat
    return value


class NOBlockConfig:

    def __init__(self, 
                 block_type: str,
                 is_encoder_only: bool,
                 is_decoder_only: bool,
                 neural_operator_type: int|list,
                 global_levels: int|list, 
                 model_dims_out: int|list,
                 n_params: List[int],
                 att_block_types_encode: List[bool],
                 global_params_learnable : List[bool],
                 att_block_types_decode: List[bool] = [],
                 global_params_init : List[float] = [],
                 nh_transformation: bool = False,
                 nh_inverse_transformation: bool = False,
                 att_dims: int = None,
                 bottle_neck_dims: int = None,
                 multi_grid_attention: bool=False,
                 with_res: bool = True,
                 neural_operator_type_nh: str='Normal_VM',
                 n_params_nh: List[int] = [[3,2]],
                 global_params_init_nh: List[float] = [[3.0]]):

        n_no_layers = len(global_levels)

        inputs = copy.deepcopy(locals())
        self.block_type = block_type
        self.multi_grid_attention = multi_grid_attention
        self.is_encoder_only = is_encoder_only
        self.is_decoder_only = is_decoder_only

        exceptions = ['self','block_type','multi_grid_attention','is_encoder_only','is_decoder_only']

        for input, value in inputs.items():
            if input not in exceptions:
                setattr(self, input, check_value(value, n_no_layers))
        

class QuantConfig:
    """
    Configuration class for defining quantization parameters in the VAE model.

    :param z_ch: Number of latent channels in the quantized representation.
    :param latent_ch: Number of latent channels in the bottleneck.
    :param block_type: Block type used in quantization (e.g., 'conv' or 'resnet').
    """

    def __init__(self, z_ch: int, latent_ch: int, block_type: str):
        self.z_ch = z_ch
        self.latent_ch = latent_ch
        self.block_type = block_type


class NOVAE(nn.Module):
    def __init__(self, 
                 icon_grid: str,
                 block_configs: List[NOBlockConfig],
  #               quant_config: QuantConfig,
                 nh: int=1,
                 seq_lvl_att: int=2,
                 model_dim_in: int=1,
                 model_dim_out: int=1,
                 n_head_channels:int=16,
                 n_vars_total:int=1,
                 rotate_coord_system: bool=True
                 ) -> None: 
        
                
        super().__init__()
        
        self.model_dim_in = model_dim_in
        self.model_dim_out = model_dim_out
        self.n_vars_total = n_vars_total

        global_levels = [torch.tensor(block_conf.global_levels) for block_conf in block_configs] + [torch.tensor(0).view(-1)]
        global_levels = torch.concat(global_levels).unique()
        
        self.register_buffer('global_levels', global_levels, persistent=False)

        mgrids = icon_grid_to_mgrid(xr.open_dataset(icon_grid),
                                    int(torch.tensor(global_levels).max()) + 1, 
                                    nh=nh)

        self.register_buffer('global_indices', torch.arange(mgrids[0]['coords'].shape[0]).unsqueeze(dim=0), persistent=False)
        self.register_buffer('cell_coords_global', mgrids[0]['coords'], persistent=False)
        
        # Create grid layers for each unique global level
        grid_layers = nn.ModuleDict()
        for global_level in global_levels:
            grid_layers[str(int(global_level))] = GridLayer(global_level, mgrids[global_level]['adjc_lvl'], mgrids[global_level]['adjc_mask'], mgrids[global_level]['coords'], coord_system='polar')

        n = 0
        global_level_in = 0
        # Construct blocks based on configurations
        self.encoder_only_blocks = nn.ModuleList()
        self.decoder_only_blocks = nn.ModuleList()

        for block_conf in block_configs:

            n_no_layers = len(block_conf.global_levels) 
            global_levels = block_conf.global_levels

            no_layers = []
            nh_no_layers = []
            for k in range(n_no_layers):
                
                global_level_in = 0 if block_conf.block_type != 'Stacked' or k==0 else global_level_no
                global_level_no = global_levels[k]

                no_layer = get_no_layer(block_conf.neural_operator_type[k], 
                                    grid_layers, 
                                    global_level_in, 
                                    global_level_no,
                                    n_params=block_conf.n_params[k],
                                    params_init=block_conf.global_params_init[k],
                                    params_learnable=block_conf.global_params_learnable[k],
                                    nh_projection=block_conf.nh_transformation[k],
                                    nh_backprojection=block_conf.nh_inverse_transformation[k],
                                    precompute_coordinates=True if n!=0 and n<n_no_layers else False,
                                    rotate_coordinate_system=rotate_coord_system)

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
                                    nh_backprojection=True)
                else:
                    no_layer_nh = None
                
                nh_no_layers.append(no_layer_nh)

                n+=1

            if block_conf.block_type == 'Serial':
                block = Serial_NOBlock

            elif block_conf.block_type == 'Stacked':
                block = Stacked_NOBlock

            elif block_conf.block_type == 'Parallel':
                block = Stacked_NOBlock
            

            Block = block(model_dim_in,
                        block_conf.model_dims_out,
                        no_layers,
                        n_params=block_conf.n_params,
                        att_block_types_encode=block_conf.att_block_types_encode,
                        att_block_types_decode=block_conf.att_block_types_decode,
                        n_head_channels=n_head_channels,
                        bottle_neck_dims=block_conf.bottle_neck_dims,
                        att_dims=block_conf.att_dims,
                        with_res= block_conf.with_res,
                        no_layers_nh=nh_no_layers,
                        multi_grid_attention=block_conf.multi_grid_attention)
            
            if block_conf.is_encoder_only:
                self.encoder_only_blocks.append(Block)

            elif block_conf.is_decoder_only:
                self.decoder_only_blocks.append(Block)

            else:
                self.vae_block = Block

        
   #     self.quantization = Quantization(in_ch=self.vae_block.model_dim_out_encode, z_ch=quant_config.z_ch, latent_ch=quant_config.latent_ch,
   #                                      block_type=quant_config.block_type, spatial_dim_count=2)
        
    def prepare_data(self, x, coords_input=None, coords_output=None, sampled_indices_batch_dict=None, mask=None):
        b,n = x.shape[:2]
        x = x.view(b,n,-1,self.model_dim_in)

        if mask is not None:
            mask = mask[:,:,:,:x.shape[2]]

        if sampled_indices_batch_dict is None:
            sampled_indices_batch_dict = {
                'global_cell': self.global_indices,
                'local_cell': self.global_indices,
                'sample': None,
                'sample_level': None,
                'output_indices': None}
        else:
            indices_layers = dict(zip(
                self.global_levels.tolist(),
                [self.get_global_indices_local(sampled_indices_batch_dict['sample'], 
                                               sampled_indices_batch_dict['sample_level'], 
                                               global_level) 
                                               for global_level in self.global_levels]))

        # Use global cell coordinates if none are provided
        if coords_input is None or coords_input.numel()==0:
            coords_input = self.cell_coords_global[sampled_indices_batch_dict['local_cell']].unsqueeze(dim=-2)

        if coords_output is None or coords_output.numel()==0:
            coords_output = self.cell_coords_global[sampled_indices_batch_dict['local_cell']].unsqueeze(dim=-2)

    
        return x, mask, indices_layers, coords_input, coords_output

    def encode(self, x, coords_input=None, sampled_indices_batch_dict=None, mask=None):
        
        x, mask, indices_layers, coords_input, _ = self.prepare_data(x, 
                                                                coords_input, 
                                                                sampled_indices_batch_dict=sampled_indices_batch_dict, 
                                                                mask=mask)
        
        for k, block in enumerate(self.encoder_only_blocks):
            coords_input = coords_input if k==0 else None
            x, mask = block(x, indices_layers, sampled_indices_batch_dict, mask=mask, coords_in=coords_input, coords_out=None)
   
        x = self.vae_block.encode(x, indices_layers, sampled_indices_batch_dict, mask=mask, coords_in=coords_input)

        return x


    def decode(self, x, coords_output=None, sampled_indices_batch_dict=None):
        
        _, _, indices_layers, _, coords_output = self.prepare_data(x, 
                                                        coords_output, 
                                                        sampled_indices_batch_dict=sampled_indices_batch_dict, 
                                                        mask=None)
        
        x = self.vae_block.decode(x, indices_layers, sampled_indices_batch_dict)

        for k, block in enumerate(self.decoder_only_blocks):
            
            coords_out = coords_output if k==len(self.decoder_only_blocks)-1  else None
            
            x, _ = block(x, indices_layers, sampled_indices_batch_dict, coords_out=coords_out)

        return x.view(x.shape[0], x.shape[1], -1)


    def forward(self, x, coords_input, coords_output, sampled_indices_batch_dict=None, mask=None):

        x = self.encode(x, coords_input, sampled_indices_batch_dict=sampled_indices_batch_dict, mask=mask)

        x = self.decode(x, coords_output, sampled_indices_batch_dict=sampled_indices_batch_dict)

        return x

    
    def get_global_indices_local(self, batch_sample_indices, sampled_level_fov, global_level):

        global_indices_sampled  = self.global_indices.view(-1, 4**sampled_level_fov[0])[batch_sample_indices]
        global_indices_sampled = global_indices_sampled.view(global_indices_sampled.shape[0], -1, 4**global_level)[:,:,0]   
        
        return global_indices_sampled // 4**global_level
    

def get_no_layer(neural_operator_type, 
                 grid_layers, 
                 global_level_in, 
                 global_level_no,
                 n_params=[],
                 params_init=[],
                 params_learnable=[],
                 nh_projection=False,
                 nh_backprojection=False,
                 precompute_coordinates=True,
                 rotate_coordinate_system=True,
                 nh=1):
    
    #if global_level_in==global_level_no:
    #    if nh==1:
    #        n_params = [3,1]
    #        neural_operator_type = 'Normal_VM'

    #    elif nh==2:
    #        n_params = [3,2]
    #        neural_operator_type = 'Normal'

    if neural_operator_type == 'Normal_VM':
        no_layer = Normal_VM_NoLayer(grid_layers,
                            global_level_in,
                            global_level_no,
                            n_phi=n_params[0],
                            n_dist=n_params[1],
                            kappa_init=params_init[0],
                            kappa_learnable=params_learnable[0],
                            dist_learnable=params_learnable[1],
                            sigma_learnable=params_learnable[2],
                            nh_projection=nh_projection,
                            nh_backprojection=nh_backprojection,
                            precompute_coordinates=precompute_coordinates,
                            rotate_coord_system=rotate_coordinate_system)
                    
    elif neural_operator_type == 'Normal':
        no_layer = Normal_NoLayer(grid_layers,
                            global_level_in,
                            global_level_no,
                            n_dist_lon=n_params[0],
                            n_dist_lat=n_params[1],
                            dist_learnable=params_learnable[0],
                            sigma_learnable=params_learnable[1],
                            nh_projection=nh_projection,
                            nh_backprojection=nh_backprojection,
                            precompute_coordinates=precompute_coordinates,
                            rotate_coord_system=rotate_coordinate_system)
        
    elif neural_operator_type == 'FT':
        no_layer = FT_NOLayer(grid_layers,
                            global_level_in,
                            global_level_no,
                            n_freq_lon=n_params[0],
                            n_freq_lat=n_params[1],
                            freq_learnable=params_learnable[0],
                            nh_projection=nh_projection,
                            nh_backprojection=nh_backprojection,
                            precompute_coordinates=precompute_coordinates,
                            rotate_coord_system=rotate_coordinate_system)
    
    return no_layer