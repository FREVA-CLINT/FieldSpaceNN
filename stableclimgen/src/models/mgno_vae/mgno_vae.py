import torch
import torch.nn as nn

import copy
from typing import List, Dict
import omegaconf
from stableclimgen.src.modules.vae.quantization import Quantization

from ...modules.icon_grids.grid_layer import GridLayer
from ...modules.neural_operator.no_blocks import UNet_NOBlock

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
                 block_configs: List[NOBlockConfig],
                 quant_config: QuantConfig,
                 model_dim_in: int=1,
                 model_dim_out: int=1,
                 n_head_channels:int=16,
                 n_vars_total:int=1,
                 rotate_coord_system: bool=True,
                 p_dropout=0.
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
        
        # Create grid layers for each unique global level
        grid_layers = nn.ModuleDict()
        for global_level in global_levels:
            grid_layers[str(int(global_level))] = GridLayer(global_level, mgrids[global_level]['adjc_lvl'], mgrids[global_level]['adjc_mask'], mgrids[global_level]['coords'], coord_system='polar')

        n_no_layers_total = len((torch.concat(global_levels_tot)))

        n = 0
        global_level_in = 0
        # Construct blocks based on configurations
        self.encoder_only_blocks = nn.ModuleList()
        self.decoder_only_blocks = nn.ModuleList()

        for block_idx, block_conf in enumerate(block_configs):

            n_no_layers = len(block_conf.global_levels) 
            global_levels = block_conf.global_levels

            model_d_in = model_dim_in if block_idx==0 else Block.model_dim_out
            n_no_layers = len(block_conf.global_levels) 
            global_levels = block_conf.global_levels

            for k in range(n_no_layers):
                                    
                global_level_in = 0 if k==0 else global_level_no
                global_level_no = global_levels[k]

                layer_settings = block_conf.layer_settings[k]
                layer_settings['global_level_in'] = global_level_in
                layer_settings['global_level_out'] = global_level_no
                layer_settings['grid_layer_in'] = grid_layers[str(global_level_in)]
                layer_settings['grid_layer_no'] = grid_layers[str(global_level_no)]
                layer_settings['precompute_coordinates'] = True if n!=0 and n<n_no_layers_total else False
                layer_settings['rotate_coordinate_system'] = rotate_coord_system

                n+=1

            Block = UNet_NOBlock(model_d_in,
                        None if block_idx < len(block_configs)-1 else model_dim_out,
                        block_conf.layer_settings,
                        n_head_channels=n_head_channels,
                        p_dropout=p_dropout,
                        skip_mode=block_conf.skip_mode,
                        global_res=block_conf.global_res)
            
            if block_conf.is_encoder_only:
                self.encoder_only_blocks.append(Block)

            elif block_conf.is_decoder_only:
                self.decoder_only_blocks.append(Block)

            else:
                self.vae_block = Block

      if quant_config:
          enc_params = [block.x_dims for block in self.vae_block.NO_Blocks]
          quant_no_block = self.vae_block.NO_Blocks[-1]
          quant_in_ch = int(torch.tensor(quant_no_block.x_dims).prod())
          self.quantization = Quantization(quant_in_ch, quant_config.latent_ch, quant_config.block_type, 1,
                                           **quant_config.sub_confs,
                                           grid_layer=quant_no_block.no_layer.grid_layers[str(quant_no_block.no_layer.global_level_no)],
                                           rotate_coord_system=rotate_coord_system,
                                           n_params=enc_params,
                                           n_head_channels=quant_config.n_head_channels)
          if self.quantization.distribution == "gaussian":
              self.noise_gamma = torch.nn.Parameter(torch.ones(quant_config.latent_ch[-1]) * 1E-6)


    def prepare_data(self, x, coords_input=None, coords_output=None, indices_sample=None, mask=None):
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
    
        return x, mask, indices_sample, coords_input, coords_output



    def encode(self, x, coords_input=None, indices_sample=None, mask=None, emb=None):
        
        x, mask, indices_sample, coords_input, _ = self.prepare_data(x, 
                                                                coords_input, 
                                                                indices_sample=indices_sample, 
                                                                mask=mask)
        
        for k, block in enumerate(self.encoder_only_blocks):
            coords_input = coords_input if k==0 else None
            x, mask = block(x, coords_in=coords_input, indices_sample=indices_sample, mask=mask, emb=emb)

        x = self.vae_block.encode(x, coords_in=coords_input, indices_sample=indices_sample, mask=mask, emb=emb)[0]

        if hasattr(self, "quantization"):
            x, mask = x.unsqueeze(dim=1), mask.unsqueeze(dim=1)
            x = self.quantization.quantize(x, indices_sample=indices_sample, emb=emb)
            x = x.squeeze(dim=1)
            posterior = self.quantization.get_distribution(x)
            return posterior
        else:
            return x



    def decode(self, x, coords_output=None, indices_sample=None, emb=None):
        
        _, _, indices_sample, _, coords_output = self.prepare_data(x, 
                                                    coords_output, 
                                                    indices_sample=indices_sample)

        if hasattr(self, "quantization"):
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

        if hasattr(self, "quantization"):
            if self.quantization.distribution == "gaussian":
                noise = torch.randn_like(posterior.mean) * self.noise_gamma
            else:
                noise = None
            z = posterior.sample(noise)
        else:
            z = posterior

        dec = self.decode(z, coords_output, indices_sample=indices_sample, emb=emb)

        return dec, posterior

    
    def get_global_indices_local(self, batch_sample_indices, sampled_level_fov, global_level):

        global_indices_sampled  = self.global_indices.view(-1, 4**sampled_level_fov[0])[batch_sample_indices]
        global_indices_sampled = global_indices_sampled.view(global_indices_sampled.shape[0], -1, 4**global_level)[:,:,0]   
        
        return global_indices_sampled // 4**global_level