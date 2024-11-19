from typing import Optional, Union, List, Tuple

import torch
import torch.nn as nn
import xarray as xr

from ...utils.grid_utils_icon import icon_grid_to_mgrid

from ...modules.distributions.distributions import DiagonalGaussianDistribution
from ...modules.neural_operator.neural_operator import NoLayer
from ...modules.icon_grids.icon_grids import GridLayer

class NOVAEBlockConfig:
    """
    Configuration class for defining the parameters of VAE blocks.

    :param depth: Number of layers in the block.
    :param block_type: Type of block (e.g., 'conv' or 'resnet').
    :param ch_mult: Channel multiplier for the block; can be an int or list of ints.
    :param blocks: List of block configurations specific to the block type.
    :param enc: Boolean indicating if this is an encoder block. Default is False.
    :param dec: Boolean indicating if this is a decoder block. Default is False.
    """

    def __init__(self, block_type: str,
                 global_level_out: int,
                 model_dim_out: int, 
                 n_sigma: int,
                 n_dist: int=1,
                 n_phi: int=1,
                 phi_attention: bool = False,
                 dist_attention: bool = False,
                 sigma_attention: bool = False,
                 dist_learnable: bool = True,
                 sigma_learnable: bool = True,
                 use_von_mises: bool = True,
                 kappa_init: float = 1.,
                 with_mean_res: bool = True,
                 with_channel_res: bool = False,
                 ref_layer_in: bool = False,
                 nh_attention: bool = False,
                 enc: bool = False,
                 dec: bool = False,
                 quant: bool = False):

        self.block_type =block_type
        self.global_level_out = global_level_out
        self.block_type = block_type
        self.model_dim_out = model_dim_out
        self.n_sigma = n_sigma
        self.n_dist = n_dist
        self.n_phi = n_phi
        self.phi_attention = phi_attention
        self.dist_attention = dist_attention
        self.sigma_attention = sigma_attention
        self.dist_learnable = dist_learnable
        self.sigma_learnable = sigma_learnable
        self.use_von_mises = use_von_mises
        self.with_mean_res = with_mean_res
        self.with_channel_res = with_channel_res
        self.kappa_init = kappa_init
        self.ref_layer_in = ref_layer_in
        self.enc = enc
        self.dec = dec
        self.quant = quant
        self.nh_attention=nh_attention


class NOVAE(nn.Module):
    """
    Variational Autoencoder (VAE) model using a configurable encoder-decoder architecture.

    :param init_in_ch: Initial input channel count.
    :param final_out_ch: Final output channel count.
    :param block_configs: List of VAEBlockConfig instances defining encoder/decoder blocks.
    :param quant_config: QuantConfig instance defining quantization configuration.
    :param model_channels: Number of model channels (default 64).
    :param embed_dim: Optional embedding dimension.
    :param patch_emb_type: Patch embedding type ("conv" or "linear").
    :param patch_emb_size: Patch embedding size (tuple of dimensions).
    :param patch_emb_kernel: Kernel size for patch embedding.
    :param concat_cond: Whether to concatenate conditional data.
    :param spatial_dim_count: Number of spatial dimensions (2 or 3).
    """

    def __init__(self,
                 icon_grid: str,
                 block_configs: List[NOVAEBlockConfig],
                 model_dim_in: int = 1,
                 model_dim_out: int = 1,
                 n_vars_total: int = 1,
                 nh:int = 1,
                 rotate_coord_system:bool = True,
                 ):
        super().__init__()

        self.enc_blocks, self.dec_blocks = nn.ModuleList(), nn.ModuleList()

        self.model_dim_in = model_dim_in
        self.model_dim_out = model_dim_out
        self.n_vars_total = n_vars_total

        global_levels = torch.tensor([block_conf.global_level_out for block_conf in block_configs]).unique()
        self.register_buffer('global_levels', global_levels, persistent=False)

        mgrids = icon_grid_to_mgrid(xr.open_dataset(icon_grid),
                                    int(torch.tensor(global_levels).max()) + 1, 
                                    nh=nh)

        self.register_buffer('global_indices', torch.arange(mgrids[0]['coords'].shape[1]).unsqueeze(dim=0), persistent=False)
        self.register_buffer('cell_coords_global', mgrids[0]['coords'], persistent=False)
        
        # Create grid layers for each unique global level
        grid_layers = nn.ModuleDict()
        for global_level in global_levels:
            grid_layers[str(int(global_level))] = GridLayer(global_level, mgrids[global_level]['adjc_lvl'], mgrids[global_level]['adjc_mask'], mgrids[global_level]['coords'], coord_system='polar')
        

        model_dim_in = model_dim_in
        global_level_in = 0
        # Construct blocks based on configurations
        for block_conf in block_configs:
            if block_conf.block_type == 'no_layer':
                
                block_conf['rotate_coord_system'] = rotate_coord_system
                no_layer = NoLayer(grid_layers,
                        global_level_in,
                        block_conf.global_level_out,
                        model_dim_in,
                        block_conf.model_dim_out,
                        kernel_settings=block_conf)
                
                if block_conf.enc:
                    self.enc_blocks.append(no_layer)
                elif block_conf.dec:
                    self.dec_blocks.append(no_layer)
                    

                model_dim_in = block_conf.model_dim_out
                global_level_in = block_conf.global_level_out
      

    def encode(self, x: torch.Tensor, coords_input, sampled_indices_batch_dict=None, mask=None) -> DiagonalGaussianDistribution:
        """
        Encodes input data to latent space, returning a posterior distribution.

        :param x: Input tensor.
        :return: DiagonalGaussianDistribution posterior distribution.
        """
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

        for k, block in enumerate(self.enc_blocks):
            
            if k ==0:
                x, mask = block(
                            x, indices_layers=indices_layers, sample_dict=sampled_indices_batch_dict, mask=mask, coordinates=coords_input
                        )
            else:
                x, mask = block(
                            x, indices_layers=indices_layers, sample_dict=sampled_indices_batch_dict, mask=mask
                        )
               
        posterior = DiagonalGaussianDistribution(x)
        return posterior

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes latent variable z back to data space.

        :param z: Latent tensor.
        :return: Decoded tensor in data space.
        """
        for block in self.dec_blocks:

            x, mask = block(
                        x, indices_layers=indices_layers, sample_dict=sampled_indices_batch_dict, mask=mask
                    )
            
        dec = self.decoder(z)
        return dec

    def forward(self, x: torch.Tensor,
                sample_posterior: bool = True) -> Tuple[torch.Tensor, DiagonalGaussianDistribution]:
        """
        Forward pass through the VAE, encoding and decoding the input.

        :param x: Input tensor.
        :param embeddings: Optional embeddings for conditional inputs.
        :param mask_data: Optional mask data for attention.
        :param cond_data: Optional conditional data for input.
        :param coords: Optional coordinates for spatial embedding.
        :param sample_posterior: Boolean flag to sample from posterior distribution.
        :return: Tuple of reconstructed tensor and posterior distribution.
        """
        # Define output shape for reconstruction
        out_shape = x.shape[1:-2]

        b,n = x.shape[:2]
        x = x.view(b,n,-1,self.model_dim_in)

        posterior = self.encode(x)
        z = posterior.sample() if sample_posterior else posterior.mode()
        dec = self.decode(z)
        return self.out(dec, out_shape=out_shape), posterior


    def get_global_indices_local(self, batch_sample_indices, sampled_level_fov, global_level):

        global_indices_sampled  = self.global_indices.view(-1, 4**sampled_level_fov[0])[batch_sample_indices]
        global_indices_sampled = global_indices_sampled.view(global_indices_sampled.shape[0], -1, 4**global_level)[:,:,0]   
        
        return global_indices_sampled // 4**global_level