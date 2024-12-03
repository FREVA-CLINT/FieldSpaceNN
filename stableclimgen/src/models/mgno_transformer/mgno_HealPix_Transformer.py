import torch
import torch.nn as nn
import torch.nn.functional as F

import xarray as xr
import omegaconf

from ...utils.grid_utils_healpix import healpix_grid_to_mgrid
from ...utils.grid_utils_icon import icon_grid_to_mgrid

from .mg_network import MultiGridBlock
from ...modules.icon_grids.grid_layer import GridLayer



def check_value(value, n_repeat):
    if not isinstance(value, list) and not isinstance(value, omegaconf.listconfig.ListConfig):
        value = [value]*n_repeat
    return value



class HealPixTransformer(nn.Module):
    def __init__(self,
                 reference_grid_level,
                 global_levels_block_encoder: list,
                 model_dims_encoder: list,
                 mg_encoder_simul: list | bool,
                 mg_encoder_n_sigma: list | int,
                 mg_decoder_n_sigma: list | int,
                 mg_encoder_n_dist: list | int=1,
                 mg_encoder_n_phi: list | int=1,
                 mg_encoder_phi_attention: list | bool = False,
                 mg_encoder_dist_attention: list | bool = False,
                 mg_encoder_sigma_attention: list | bool = False,
                 mg_decoder_n_dist: list | int=1,
                 mg_decoder_n_phi: list | int=1,
                 mg_decoder_phi_attention: list | bool = False,
                 mg_decoder_dist_attention: list | bool = False,
                 mg_decoder_sigma_attention: list | bool = False,
                 mg_decoder_nh_projection : list | bool = True,
                 global_levels_block_decoder: list=[],
                 model_dims_decoder: list =[],
                 dist_learnable: list | bool = True,
                 sigma_learnable: list | bool = True,
                 kappa_learnable: list | bool = True,
                 use_von_mises: list | bool = True,
                 with_mean_res: list | bool = True,
                 with_channel_res: list | bool = False,
                 kappa_init: list | float = 1.,
                 mg_spa_method: list | str = None,
                 mg_spa_min_lvl: list | str = None,
                 mg_encoder_kernel_settings_for_spa: bool = True,
                 mg_attention_chunks:int=2,
                 nh: int=1,
                 seq_lvl_att: int=2,
                 model_dim_in: int=1,
                 model_dim_out: int=1,
                 n_head_channels:int=16,
                 pos_emb_calc: str='cartesian_km',
                 n_vars_total:int=1,
                 rotate_coord_system: bool=True
                 ) -> None: 
        
        """
        Initialize the ICON_Transformer class.

        :param icon_grid: Path to the ICON grid file.
        :param global_levels_block_encoder: List of grid levels for the encoder blocks.
        :param model_dims_encoder: List of model dimensions for the encoder blocks.
        :param mg_encoder_simul: Whether to perform simultaneous multi-grid encoding.
        :param mg_encoder_n_sigma: Number of sigma (scale) components for the encoder.
        :param mg_decoder_n_sigma: Number of sigma (scale) components for the decoder.
        :param mg_encoder_n_dist: Number of distance components for the encoder.
        :param mg_encoder_n_phi: Number of phi (angular) components for the encoder.
        :param mg_encoder_phi_attention: Whether to use phi channel attention in the encoder.
        :param mg_encoder_dist_attention: Whether to use distance channel attention in the encoder.
        :param mg_encoder_sigma_attention: Whether to use sigma channel attention in the encoder.
        :param mg_decoder_n_dist: Number of distance components for the decoder.
        :param mg_decoder_n_phi: Number of phi (angular) components for the decoder.
        :param mg_decoder_phi_attention: Whether to use phi channel attention in the decoder.
        :param mg_decoder_dist_attention: Whether to use distance channel attention in the decoder.
        :param mg_decoder_sigma_attention: Whether to use sigma channel attention in the decoder.
        :param mg_decoder_nh_projection: Whether to use nh (neighborhood) projection in the decoder.
        :param global_levels_block_decoder: List of grid levels for the decoder blocks.
        :param model_dims_decoder: List of model dimensions for the decoder blocks.
        :param dist_learnable: Whether distance components are learnable.
        :param sigma_learnable: Whether sigma components are learnable.
        :param use_von_mises: Whether to use the von Mises distribution for angular weighting.
        :param with_mean_res: Whether to use mean residual connections.
        :param with_channel_res: Whether to use channel-wise residual connections.
        :param kappa_init: Initial value for kappa (angular concentration parameter).
        :param mg_spa_method: Spatial method used in the multi-grid encoding.
        :param mg_spa_min_lvl: Minimum level for spatial processing.
        :param mg_encoder_kernel_settings_for_spa: Whether the kernel settings of the encoder should be used for multi-grid spatial attention
        :param nh: Number of neighborhoods.
        :param seq_lvl_att: Number of sequential level attention layers.
        :param model_dim_in: Input model dimensionality. Set to 1 if model runs in variable-independent mode
        :param model_dim_out: Output model dimensionality. Set to 1 if model runs in variable-independent mode
        :param n_head_channels: Number of channels per attention head.
        :param pos_emb_calc: Method for position embedding calculation.
        :param n_vars_total: Total number of variables to handle. Set to 1 if model runs in variable-dependent mode
        :param rotate_coord_system: Whether to rotate the coord system with respect to target points.

        """
                
        super().__init__()
        
        # Flatten and combine encoder and decoder grid levels into a single tensor
        global_levels_encode_flat = torch.concat([torch.tensor(l) for l in global_levels_block_encoder])
        global_levels_decode_flat = torch.concat([torch.tensor(l) for l in global_levels_block_decoder]) if len(global_levels_block_decoder)>0 else global_levels_encode_flat
        global_levels = torch.concat((global_levels_encode_flat, global_levels_decode_flat, torch.tensor(0).view(-1))).unique()
        self.register_buffer('global_levels', global_levels, persistent=False)

        # Create multi-grid structures using the provided ICON grid
        mgrids = healpix_grid_to_mgrid(reference_grid_level, int(torch.tensor(global_levels).max()) + 1, nh=nh)
        self.coord_system = "polar" if "polar" in  pos_emb_calc else "cartesian"
        
        # Store global cell indices and cell coordinates for reference
        self.register_buffer('global_indices', torch.arange(mgrids[0]['coords'].shape[1]).unsqueeze(dim=0), persistent=False)
        self.register_buffer('cell_coords_global', mgrids[0]['coords'], persistent=False)
        
        # Create grid layers for each unique global level
        grid_layers = nn.ModuleDict()
        for global_level in global_levels:
            grid_layers[str(int(global_level))] = GridLayer(global_level, mgrids[global_level]['adjc_lvl'], mgrids[global_level]['adjc_mask'], mgrids[global_level]['coords'], coord_system=self.coord_system)
        
    
        n_blocks = len(global_levels_block_encoder)

        # Ensure all configurations are properly assigned for each block
        model_dims_encoder = check_value(model_dims_encoder, n_blocks)
        mg_encoder_simul = check_value(mg_encoder_simul, n_blocks)
        mg_spa_method = check_value(mg_spa_method, n_blocks)
        mg_spa_min_lvl = check_value(mg_spa_min_lvl, n_blocks)
        mg_encoder_n_sigma = check_value(mg_encoder_n_sigma, n_blocks)
        mg_encoder_n_dist = check_value(mg_encoder_n_dist, n_blocks)
        mg_encoder_n_phi = check_value(mg_encoder_n_phi, n_blocks)
        mg_encoder_phi_attention = check_value(mg_encoder_phi_attention, n_blocks)
        mg_encoder_dist_attention = check_value(mg_encoder_dist_attention, n_blocks)
        mg_encoder_sigma_attention = check_value(mg_encoder_sigma_attention, n_blocks)

        mg_decoder_n_sigma = check_value(mg_decoder_n_sigma, n_blocks)
        mg_decoder_n_dist = check_value(mg_decoder_n_dist, n_blocks)
        mg_decoder_n_phi = check_value(mg_decoder_n_phi, n_blocks)
        mg_decoder_n_dist = check_value(mg_decoder_n_dist, n_blocks)
        mg_decoder_phi_attention = check_value(mg_decoder_phi_attention, n_blocks)
        mg_decoder_dist_attention = check_value(mg_decoder_dist_attention, n_blocks)
        mg_decoder_sigma_attention = check_value(mg_decoder_sigma_attention, n_blocks)
        mg_decoder_nh_projection = check_value(mg_decoder_nh_projection, n_blocks)

        dist_learnable = check_value(dist_learnable, n_blocks)
        sigma_learnable = check_value(sigma_learnable, n_blocks)
        kappa_learnable = check_value(kappa_learnable, n_blocks)
        use_von_mises = check_value(use_von_mises, n_blocks)
        with_mean_res = check_value(with_mean_res, n_blocks)
        with_channel_res = check_value(with_channel_res, n_blocks)
        kappa_init = check_value(kappa_init, n_blocks)

        self.model_dim_in = model_dim_in

        # If decoder blocks are not provided, create default settings
        if len(global_levels_block_decoder)==0:
            global_levels_block_decoder = list([[int(k)] for k in torch.tensor(global_levels_block_encoder)[:,0]])
            global_levels_block_decoder[-1]=[0]

            model_dims_decoder = list([[int(k)] for k in torch.tensor(model_dims_encoder)[:,0]])
            #model_dims_decoder[-1] = model_dim_out
        else:
            global_levels_block_decoder = check_value(global_levels_block_decoder, n_blocks)
            model_dims_decoder = check_value(model_dims_decoder, n_blocks)


        # Initialize Multi-Grid Blocks
        self.MGBlocks = nn.ModuleList()
        for k in range(n_blocks):
            global_level_in = 0 if k==0 else global_levels_block_decoder[k-1][-1]
            model_dim_in = model_dim_in if k==0 else model_dims_decoder[k-1][-1]
            global_levels_encode = global_levels_block_encoder[k]
            global_levels_decode = global_levels_block_decoder[k]
            model_dims_encode = model_dims_encoder[k] 
            model_dims_decode = model_dims_decoder[k]

            #if k==n_blocks-1:
                #if global_levels_decode[-1]!=0:
                #    global_levels_decode.append(0)
                #    model_dims_decode.append(model_dim_out)
                #else:
                #    model_dims_decode[-1]=model_dim_out

            no_layer_encoder = {'n_sigma': mg_encoder_n_sigma[k],
                                'n_dists': mg_encoder_n_dist[k],
                                'n_phi': mg_encoder_n_phi[k],
                                'sigma_att': mg_encoder_sigma_attention[k],
                                'dist_att': mg_encoder_dist_attention[k],
                                'phi_att': mg_encoder_phi_attention[k],
                                'n_sigma': mg_encoder_n_sigma[k],
                                'nh_projection': False,
                                'dists_learnable': dist_learnable[k],
                                'sigma_learnable': sigma_learnable[k],
                                'kappa_learnable': kappa_learnable[k],
                                'use_von_mises': use_von_mises[k],
                                'with_mean_res': with_mean_res[k],
                                'with_channel_res': with_channel_res[k],
                                'kappa_init': kappa_init[k],
                                'rotate_coord_system': rotate_coord_system}

            no_layer_decoder = {'n_sigma': mg_decoder_n_sigma[k],
                                'n_dists': mg_decoder_n_dist[k],
                                'n_phi': mg_decoder_n_phi[k],
                                'sigma_att': mg_decoder_sigma_attention[k],
                                'dist_att': mg_decoder_dist_attention[k],
                                'phi_att': mg_decoder_phi_attention[k],
                                'n_sigma': mg_decoder_n_sigma[k],
                                'nh_projection': mg_decoder_nh_projection[k],
                                'dists_learnable': dist_learnable[k],
                                'sigma_learnable': sigma_learnable[k],
                                'kappa_learnable': kappa_learnable[k],
                                'use_von_mises': use_von_mises[k],
                                'with_mean_res': with_mean_res[k],
                                'with_channel_res': with_channel_res[k],
                                'kappa_init': kappa_init[k],
                                'rotate_coord_system': rotate_coord_system}

            no_layer_processing = no_layer_encoder if mg_encoder_kernel_settings_for_spa else no_layer_decoder

            # Add a Multi-Grid Block to the model
            self.MGBlocks.append(MultiGridBlock(grid_layers,
                                                global_level_in,
                                                global_levels_encode,
                                                global_levels_decode, 
                                                model_dim_in,
                                                model_dims_encode,
                                                model_dims_decode, 
                                                n_vars_total,
                                                encoder_no_layer_settings = no_layer_encoder,
                                                decoder_no_layer_settings = no_layer_decoder,
                                                encoder_simul = mg_encoder_simul[k],
                                                processing_method = mg_spa_method[k],
                                                processing_min_lvl = mg_spa_min_lvl[k],
                                                processing_no_layer_settings = no_layer_processing,
                                                seq_level_attention = seq_lvl_att, 
                                                nh = nh,
                                                n_head_channels=n_head_channels,
                                                mg_attention_chunks=mg_attention_chunks,
                                                pos_emb_calc='cartesian_km',
                                                emb_table_bins=16,
                                                first_block=True if k==0 else False,
                                                output_dim=model_dim_out if k==n_blocks-1 else None))
        
        


    def forward(self, x, coords_input, coords_output, sampled_indices_batch_dict=None, mask=None):

        """
        Forward pass for the ICON_Transformer model.

        :param x: Input tensor of shape (batch_size, num_cells, input_dim).
        :param coords_input: Input coordinates for x
        :param coords_output: Output coordinates for position embedding.
        :param sampled_indices_batch_dict: Dictionary of sampled indices for regional models.
        :param mask: Mask for dropping cells in the input tensor.
        :return: Output tensor of shape (batch_size, num_cells, output_dim).
        """

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
        if coords_input.numel()==0:
            coords_input = self.cell_coords_global[:,sampled_indices_batch_dict['local_cell']].unsqueeze(dim=-1)

        if coords_output.numel()==0:
            coords_output = self.cell_coords_global[:,sampled_indices_batch_dict['local_cell']].unsqueeze(dim=-1)

        # Pass through each Multi-Grid Block
        for k, multi_grid_block in enumerate(self.MGBlocks):
            
            coords_in = coords_input if k==0 else None
            coords_out = coords_output if k==len(self.MGBlocks)-1  else None
            
            # Process input through the block
            x, mask = multi_grid_block(x, indices_layers, sampled_indices_batch_dict, mask=mask, coords_in=coords_in, coords_out=coords_out)

        # Reshape output to match the expected dimensions
        x = x.view(b,n,-1)
        return x

    
    def get_global_indices_local(self, batch_sample_indices, sampled_level_fov, global_level):

        global_indices_sampled  = self.global_indices.view(-1, 4**sampled_level_fov[0])[batch_sample_indices]
        global_indices_sampled = global_indices_sampled.view(global_indices_sampled.shape[0], -1, 4**global_level)[:,:,0]   
        
        return global_indices_sampled // 4**global_level
    
