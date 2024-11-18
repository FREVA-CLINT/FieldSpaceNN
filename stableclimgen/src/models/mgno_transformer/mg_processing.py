import torch
import torch.nn as nn

from .icon_spatial_attention import IconSpatialAttention

class ProcessingLayer(nn.Module):
    """
    A processing layer module designed to operate on hierarchical grid layers, applying
    spatial attention mechanisms to transform feature representations at different 
    global levels.

    This class utilizes spatial attention mechanisms to focus on specific parts of the 
    input data across different hierarchical levels in the grid, which is useful in 
    scenarios like geospatial modeling or structured grid-based data analysis.

    Attributes:
        global_levels (list): List of global levels indicating the hierarchy of grid layers.
        mode (str): The kernel function mode used for the attention, e.g., 'spatial' or 'nh'.
        processing_layers (nn.ModuleDict): A dictionary of processing layers, each associated
            with a global level, for performing spatial attention.
        gammas (nn.ParameterDict): A dictionary of learnable gamma parameters for tuning
            the influence of each layer.
        grid_layers (dict): The dictionary containing grid layer configurations.
    """

    def __init__(self, 
                 processing_method: str,
                 grid_layers: dict,
                 global_levels: list, 
                 model_dims: list,
                 seq_level_attention: bool,
                 nh: int = 1,
                 n_head_channels: int = 16,
                 pos_emb_calc: str = 'cartesian_km',
                 emb_table_bins: int = 16,
                 rotate_coord_system=True) -> None:
        """
        Initializes the processing_layer class with spatial attention mechanisms.

        :param grid_layers: A dictionary containing grid layer configurations.
        :param global_levels: List of global levels to process data at.
        :param model_dims: List specifying the dimensionality of features at each grid level.
        :param seq_level_attention: Boolean indicating if sequence-level attention is used.
        :param n_vars_total: The total number of variables in the input data.
        :param nh: Number of heads for attention mechanisms.
        :param n_head_channels: Number of channels per head in multi-head attention.
        :param pos_emb_calc: Method for position embedding calculation, e.g., 'cartesian_km'.
        :param emb_table_bins: Number of bins used for embedding table.
        :param kernel_fcn: Kernel function type for attention, e.g., 'spatial' or 'nh'.
        :param kernel_dim: Dimensionality of the kernel for attention.
        :param n_chunks: Number of chunks for dividing the data, if needed.
        """
        super().__init__()

        self.global_levels = global_levels
        self.processing_layers = nn.ModuleDict()
        self.gammas = nn.ParameterDict()

        # Initialize processing layers for each global level
        for k, global_level in enumerate(global_levels):
            global_level = str(int(global_level))

            nh_attention = 'nh' in processing_method
            self.processing_layers[global_level] = IconSpatialAttention(
                grid_layers, 
                global_level, 
                int(model_dims[k]),
                n_head_channels,
                seq_level_attention,
                nh=nh,
                pos_emb_calc=pos_emb_calc,
                emb_table_bins=emb_table_bins, 
                nh_attention=nh_attention, 
                continous_pos_embedding=True,
                rotate_coord_system=rotate_coord_system
            )

        self.grid_layers = grid_layers

    def forward(self, x_levels: dict, indices_layers: dict, batch_dict: dict, drop_masks_levels: dict = None):
        """
        Forward pass to process data at multiple global levels using spatial attention.

        :param x_levels: A dictionary containing tensors for each input grid level.
        :param indices_layers: A dictionary with indices for the grid layers.
        :param batch_dict: Dictionary containing batch-related information for processing.
        :param drop_masks_levels: Optional dictionary of drop masks for each grid level.
        :return: A tuple containing:
            - Updated x_levels: A dictionary of processed tensors at each global level.
            - Updated drop_masks_levels: A dictionary of updated drop masks.
        """
        # Iterate over each global level and apply spatial attention
        for k, global_level in enumerate(self.global_levels):
            global_level = int(global_level)

            # Apply the processing layer for the current global level
            x, mask = self.processing_layers[str(global_level)](
                x_levels[global_level], 
                drop_masks_levels[global_level], 
                indices_layers, 
                batch_dict
            )

            # Update the output tensors and masks for the current level
            x_levels[global_level] = x
            drop_masks_levels[global_level] = mask

        return x_levels, drop_masks_levels