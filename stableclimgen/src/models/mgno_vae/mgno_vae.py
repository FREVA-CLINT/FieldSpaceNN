import torch
import torch.nn as nn

import copy
from typing import List, Dict
import omegaconf
from stableclimgen.src.modules.vae.quantization import Quantization

from ...modules.icon_grids.grid_layer import GridLayer
from ..mgno_transformer.mgno_serial_block import Serial_NOBlock

from ..mgno_transformer.mgno_block_confs import NOBlockConfig


class QuantConfig:
    """
    Configuration class for defining quantization parameters in the VAE model.

    :param z_ch: Number of latent channels in the quantized representation.
    :param latent_ch: Number of latent channels in the bottleneck.
    :param block_type: Block type used in quantization (e.g., 'conv' or 'resnet').
    """

    def __init__(self, latent_ch: List[int], n_head_channels: List[int], block_type: str, sub_confs: dict):
        self.latent_ch = latent_ch
        self.block_type = block_type
        self.sub_confs = sub_confs
        self.n_head_channels = n_head_channels


class MGNO_VAE(nn.Module):
    def __init__(self, 
                 mgrids,
                 encoder_block_configs: List[NOBlockConfig],
                 decoder_block_configs: List[NOBlockConfig],
                 quant_config: QuantConfig,
                 input_dim: int=1,
                 lifting_dim: int=1,
                 output_dim: int=1,
                 n_head_channels:int=16,
                 n_vars_total:int=1,
                 rotate_coord_system: bool=True,
                 mask_as_embedding: bool=False,
                 p_dropout=0.,
                 ) -> None: 
        
                
        super().__init__()
        
        self.n_vars_total = n_vars_total

        global_levels_out_enc = [[layer_setting.get("global_level_decode", 0) 
                              for layer_setting in block_conf.layer_settings]
                              for block_conf in encoder_block_configs]
        
        global_levels_out_dec = [[layer_setting.get("global_level_decode", 0) 
                              for layer_setting in block_conf.layer_settings]
                              for block_conf in decoder_block_configs]

        global_levels_no_enc = [[layer_setting.get("global_level_no", 0) 
                              for layer_setting in block_conf.layer_settings]
                              for block_conf in encoder_block_configs]
        
        global_levels_no_dec = [[layer_setting.get("global_level_no", 0) 
                              for layer_setting in block_conf.layer_settings]
                              for block_conf in decoder_block_configs]
        
        global_levels = torch.concat((torch.tensor(global_levels_out_enc).view(-1),
                                      torch.tensor(global_levels_out_dec).view(-1),
                                      torch.tensor(global_levels_no_enc).view(-1),
                                      torch.tensor(global_levels_no_dec).view(-1),
                                      torch.tensor(0).view(-1))).unique()
        
        self.register_buffer('global_levels', global_levels, persistent=False)
        self.register_buffer('global_indices', torch.arange(mgrids[0]['coords'].shape[0]).unsqueeze(dim=0), persistent=False)
        self.register_buffer('cell_coords_global', mgrids[0]['coords'], persistent=False)
        
        self.input_dim = input_dim

        # Create grid layers for each unique global level
        grid_layers = nn.ModuleDict()
        for global_level in global_levels:
            grid_layers[str(int(global_level))] = GridLayer(global_level, mgrids[global_level]['adjc_lvl'], mgrids[global_level]['adjc_mask'], mgrids[global_level]['coords'], coord_system='polar')

        self.grid_layer_0 = grid_layers["0"]
        
        # Construct blocks based on configurations
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        
        self.lifting_layer = nn.Linear(input_dim, lifting_dim, bias=False) if lifting_dim>1 else nn.Identity()

        for block_idx, block_conf in enumerate(encoder_block_configs):
            layer_settings = block_conf.layer_settings
            model_dims_out = block_conf.model_dims_out

            block = Serial_NOBlock(
                lifting_dim,
                model_dims_out,
                grid_layers,
                layer_settings,
                rotate_coordinate_system=rotate_coord_system)
                
            
            self.encoder_blocks.append(block)
        
        output_level_encoder = block.output_level
        output_dim_encoder = model_dims_out[-1]

        for block_idx, block_conf in enumerate(decoder_block_configs):
            layer_settings = block_conf.layer_settings
            model_dims_out = block_conf.model_dims_out

            block = Serial_NOBlock(
                output_dim_encoder,
                model_dims_out,
                grid_layers,
                layer_settings,
                input_level=output_level_encoder,
                output_dim=output_dim,
                rotate_coordinate_system=rotate_coord_system,
                mask_as_embedding=mask_as_embedding)
                
            self.decoder_blocks.append(block)


        if quant_config:
            self.quantization = Quantization(output_dim_encoder, quant_config.latent_ch, quant_config.block_type, 1,
                                            **quant_config.sub_confs,
                                            grid_layer=grid_layers[str(output_level_encoder)],
                                            rotate_coord_system=rotate_coord_system,
                                            n_head_channels=quant_config.n_head_channels)


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



    def encode(self, x, coords_input=None, indices_sample=None, mask=None, emb=None):
        
        b,n,nh,nv,nc = x.shape[:5]
        x = x.view(b,n,-1,self.input_dim)
        b,n,nv,nc = x.shape[:4]

        if mask is not None:
            mask = mask.view(*x.shape[:-1])

        indices_sample, coords_input, _ = self.prepare_coords_indices(coords_input, 
                                                                indices_sample=indices_sample)
        
        x = self.lifting_layer(x)

        for k, block in enumerate(self.encoder_blocks):
            coords_input = coords_input if k==0 else None
            x, mask = block(x, coords_in=coords_input, indices_sample=indices_sample, mask=mask, emb=emb)

        if hasattr(self, "quantization"):
            if mask is not None:
                mask = mask.unsqueeze(dim=1)
            x = self.quantization.quantize(x.unsqueeze(1), indices_sample=indices_sample, emb=emb)
            x = x.squeeze(dim=1)
            posterior = self.quantization.get_distribution(x)
            return posterior
        else:
            return x


    def decode(self, x, coords_output=None, indices_sample=None, emb=None):
        
        indices_sample, _, coords_output = self.prepare_coords_indices(coords_output, 
                                                    indices_sample=indices_sample)

        if hasattr(self, "quantization"):
            x = x.unsqueeze(dim=1)
            x = self.quantization.post_quantize(x, indices_sample=indices_sample, emb=emb)
            x = x.squeeze(dim=1)

        for k, block in enumerate(self.decoder_blocks):
            
            coords_out = coords_output if k==len(self.decoder_blocks)-1  else None
            
            x, _ = block(x, coords_out=coords_out, indices_sample=indices_sample, emb=emb)

        return x.view(x.shape[0], x.shape[1], -1)


    def forward(self, x, coords_input, coords_output, indices_sample=None, mask=None, emb=None):

        posterior = self.encode(x, coords_input, indices_sample=indices_sample, mask=mask, emb=emb)

        if hasattr(self, "quantization"):
            z = posterior.sample()
        else:
            z = posterior
        dec = self.decode(z, coords_output, indices_sample=indices_sample, emb=emb)

        return dec, posterior

    
    def get_global_indices_local(self, batch_sample_indices, sampled_level_fov, global_level):

        global_indices_sampled  = self.global_indices.view(-1, 4**sampled_level_fov[0])[batch_sample_indices]
        global_indices_sampled = global_indices_sampled.view(global_indices_sampled.shape[0], -1, 4**global_level)[:,:,0]   
        
        return global_indices_sampled // 4**global_level