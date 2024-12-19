import torch
import torch.nn as nn

import copy
from typing import List, Dict
import omegaconf
from stableclimgen.src.modules.vae.quantization import Quantization
from ...modules.distributions.distributions import DiagonalGaussianDistribution

from ...modules.icon_grids.grid_layer import GridLayer
from ...modules.neural_operator.no_blocks import Stacked_NOBlock, Serial_NOBlock, Parallel_NOBlock, UNet_NOBlock

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
                 bottle_neck_dims: int=None,
                 embed_names_encode: List[List[str]] = None,
                 embed_names_decode: List[List[str]] = None,
                 embed_confs: Dict = None,
                 embed_mode: str = "sum",
                 nh_transformation: bool = False,
                 nh_inverse_transformation: bool = False,
                 att_dims: int = None,
                 multi_grid_attention: bool=False,
                 neural_operator_type_nh: str='Normal_VM',
                 n_params_nh: List[int] = [[3,2]],
                 global_params_init_nh: List[float] = [[3.0]],
                 spatial_attention_configs: dict = {}):
        

        n_no_layers = len(global_levels)

        inputs = copy.deepcopy(locals())
        self.block_type = block_type
        self.is_encoder_only = is_encoder_only
        self.is_decoder_only = is_decoder_only
        self.multi_grid_attention = multi_grid_attention
        self.bottle_neck_dims = bottle_neck_dims

        exceptions = ['self','block_type','multi_grid_attention','is_encoder_only','is_decoder_only', "bottle_neck_dims"]

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

    def __init__(self, latent_ch: List[int], block_type: str, sub_confs: dict):
        self.latent_ch = latent_ch
        self.block_type = block_type
        self.sub_confs = sub_confs


class MGNO_VAE(nn.Module):
    def __init__(self, 
                 mgrids,
                 block_configs: List[NOBlockConfig],
                 quant_config: QuantConfig,
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
                block = Parallel_NOBlock
            
            elif block_conf.block_type == 'UNet':
                block = UNet_NOBlock
            
            Block = block(model_dim_in,
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
                        spatial_attention_configs=block_conf.spatial_attention_configs)
            
            if block_conf.is_encoder_only:
                self.encoder_only_blocks.append(Block)

            elif block_conf.is_decoder_only:
                self.decoder_only_blocks.append(Block)

            else:
                self.vae_block = Block
        quant_no_block = self.vae_block.NO_Blocks[-1]
        quant_in_ch = int(quant_no_block.model_dim_in*torch.tensor(quant_no_block.n_params).prod())
        self.quantization = Quantization(quant_in_ch, quant_config.latent_ch, quant_config.block_type, 1,
                                         **quant_config.sub_confs,
                                         grid_layer=quant_no_block.no_layer.grid_layers[str(quant_no_block.no_layer.global_level_no)],
                                         rotate_coord_system=rotate_coord_system)

    def prepare_data(self, x, coords_input=None, coords_output=None, indices_sample=None, mask=None):
        b,n = x.shape[:2]
        x = x.view(b,n,-1,self.model_dim_in)

        if mask is not None:
            mask = mask[:,:,:,:x.shape[2]]

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
    
        return x, mask, indices_sample, coords_input, coords_output



    def encode(self, x, coords_input=None, indices_sample=None, mask=None, emb=None):
        
        x, mask, indices_sample, coords_input, _ = self.prepare_data(x, 
                                                                coords_input, 
                                                                indices_sample=indices_sample, 
                                                                mask=mask)
        
        for k, block in enumerate(self.encoder_only_blocks):
            coords_input = coords_input if k==0 else None
            x, mask = block(x, coords_in=coords_input, indices_sample=indices_sample, mask=mask, emb=emb)

        x, _ = self.vae_block.encode(x, coords_in=coords_input, indices_sample=indices_sample, mask=mask, emb=emb)

        x, mask = x.unsqueeze(dim=1), mask.unsqueeze(dim=1)
        x = self.quantization.quantize(x, indices_sample=indices_sample, emb=emb)
        x = x.squeeze(dim=1)
        posterior = DiagonalGaussianDistribution(x)

        return posterior



    def decode(self, x, coords_output=None, indices_sample=None, emb=None):
        
        _, _, indices_sample, _, coords_output = self.prepare_data(x, 
                                                    coords_output, 
                                                    indices_sample=indices_sample)

        x = x.unsqueeze(dim=1)
        x = self.quantization.post_quantize(x, indices_sample=indices_sample, emb=emb)
        x = x.squeeze(dim=1)

        x, _ = self.vae_block.decode(x, indices_sample=indices_sample, emb=emb)

        for k, block in enumerate(self.decoder_only_blocks):
            
            coords_out = coords_output if k==len(self.decoder_only_blocks)-1  else None
            
            x, _ = block(x, coords_out=coords_out, indices_sample=indices_sample, emb=emb)

        return x.view(x.shape[0], x.shape[1], -1)


    def forward(self, x, coords_input, coords_output, indices_sample=None, mask=None, emb=None):

        posterior = self.encode(x, coords_input, indices_sample=indices_sample, mask=mask, emb=emb)

        z = posterior.sample()

        dec = self.decode(z, coords_output, indices_sample=indices_sample, emb=emb)

        return dec, posterior

    
    def get_global_indices_local(self, batch_sample_indices, sampled_level_fov, global_level):

        global_indices_sampled  = self.global_indices.view(-1, 4**sampled_level_fov[0])[batch_sample_indices]
        global_indices_sampled = global_indices_sampled.view(global_indices_sampled.shape[0], -1, 4**global_level)[:,:,0]   
        
        return global_indices_sampled // 4**global_level