import torch
import torch.nn as nn

from .mg_encoder import MultiGridEncoder
from .mg_processing import ProcessingLayer
from .mg_decoder import MultiGridDecoder

class MultiGridBlock(nn.Module):
    def __init__(self, 
                 grid_layers,
                 global_level_in,
                 global_levels_encode,
                 global_levels_decode, 
                 model_dim_in,
                 model_dims_encode,
                 model_dims_decode, 
                 n_vars_total,
                 encoder_no_layer_settings,
                 decoder_no_layer_settings,
                 encoder_simul=False,
                 processing_method=None,
                 processing_min_lvl=2,
                 seq_level_attention=2, 
                 nh=1,
                 n_head_channels=16, 
                 pos_emb_calc='cartesian_km',
                 emb_table_bins=16,
                 first_block=False,
                 last_block=False,
                 ):
        
        """
        Initialize the MultiGridBlock class.

        :param grid_layers: ModuleDict of grid layers for multi-grid operations.
        :param global_level_in: The global level for the input data.
        :param global_levels_encode: List of global levels to be used in the encoder.
        :param global_levels_decode: List of global levels to be used in the decoder.
        :param model_dim_in: Input model dimension.
        :param model_dims_encode: List of model dimensions for the encoder layers.
        :param model_dims_decode: List of model dimensions for the decoder layers.
        :param n_vars_total: Total number of variables for processing.
        :param encoder_no_layer_settings: Settings for the encoder layers (e.g., attention types).
        :param decoder_no_layer_settings: Settings for the decoder layers (e.g., attention types).
        :param encoder_simul: Whether to perform simultaneous multi-grid encoding.
        :param processing_method: Method used for additional processing (if any).
        :param processing_min_lvl: Minimum level for applying the processing method.
        :param seq_level_attention: Number of sequential level attention layers.
        :param nh: Number of neighborhoods.
        :param n_head_channels: Number of channels per attention head.
        :param pos_emb_calc: Method for position embedding calculation.
        :param emb_table_bins: Number of bins for embedding table.
        :param output_layer: Whether this block is the final output layer.
        """

        super().__init__()      
        
        self.decomp_layer = MultiGridEncoder(grid_layers, 
                                               global_level_in, 
                                               global_levels_encode, 
                                               model_dim_in,
                                               model_dims_encode,
                                               no_layer_settings=encoder_no_layer_settings,
                                               n_head_channels=n_head_channels, 
                                               simultaneous=encoder_simul,
                                               first_encoder=first_block)


        if processing_method is not None:
            global_levels_process = torch.tensor(global_levels_encode)
            proc_idx = global_levels_process >= int(processing_min_lvl)
            global_levels_process = global_levels_process[proc_idx]
            model_dims_processing = torch.tensor(model_dims_encode)[proc_idx]

            self.processing_layer = ProcessingLayer(processing_method,
                                                    grid_layers,
                                                    global_levels_process,  
                                                    model_dims_processing,
                                                    seq_level_attention,
                                                    nh=nh,
                                                    n_head_channels=n_head_channels,
                                                    pos_emb_calc=pos_emb_calc,
                                                    emb_table_bins=emb_table_bins)


        self.mg_layer = MultiGridDecoder(grid_layers,
                                           global_levels_encode,
                                           global_levels_decode, 
                                           model_dims_encode,
                                           model_dims_decode,
                                           no_layer_settings=decoder_no_layer_settings,
                                           n_head_channels=n_head_channels,
                                           output_layer=last_block)


    def forward(self, x, indices_layers, indices_batch_dict, mask=None, coords_in=None, coords_out=None):
        """
        Forward pass for the MultiGridBlock.

        :param x: Input tensor of shape (batch_size, num_cells, input_dim).
        :param indices_layers: Dictionary containing indices for each grid layer level.
        :param indices_batch_dict: Dictionary of sampled indices for multi-grid processing.
        :param mask: Mask tensor for dropping cells in the input.
        :param coords_in: Coordinates for position embedding in the input.
        :param coords_out: Coordinates for position embedding in the output.
        :return: Tuple (x, mask) where x is the output tensor and mask is the updated mask.
        """
        x_levels, mask_levels = self.decomp_layer(x, indices_layers, drop_mask=mask, coords_in=coords_in, sample_dict=indices_batch_dict)

        if hasattr(self, 'processing_layer'):
            x_levels, mask_levels = self.processing_layer(x_levels, indices_layers, indices_batch_dict, mask_levels)

        x, mask = self.mg_layer(x_levels, indices_layers, mask_levels, indices_batch_dict, coords_out=coords_out)

        return x, mask