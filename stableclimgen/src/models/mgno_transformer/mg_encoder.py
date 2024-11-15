import torch
import torch.nn as nn

from .neural_operator import NoLayer

class MultiGridEncoder(nn.Module):
    """
    A multi-level encoder designed to operate on hierarchical grid data structures.
    The encoder aggregates data across multiple grid levels using specialized layers
    for data transformation and refinement.

    Attributes:
        grid_layers (dict): A dictionary of grid layers used for hierarchical processing.
        global_level_in (int): The starting global level of the input grid data.
        global_levels (list): A sorted list of global levels where the data will be processed.
        simultaneous (bool): If True, processes all levels simultaneously; otherwise, sequentially.
        aggregation_layers (nn.ModuleDict): A dictionary of aggregation layers used to transform data
            from one grid level to another.
    """

    def __init__(self, 
                 grid_layers: dict, 
                 global_level_in: int, 
                 global_levels_out: list, 
                 model_dim_in: int, 
                 model_dims: dict, 
                 no_layer_settings, 
                 simultaneous: bool = False,
                 n_head_channels: int = 16,
                 first_encoder=True) -> None:
        """
        Initializes the multi_grid_encoder for hierarchical data aggregation.

        :param grid_layers: Dictionary containing grid layers for different levels.
        :param global_level_in: The initial global level for input data.
        :param global_levels_out: List of global levels to process data through.
        :param model_dim_in: The input dimensionality of the model.
        :param model_dims: Dictionary specifying the output dimensionality for each level.
        :param no_layer_settings: Settings for the no_layer configuration.
        :param simultaneous: If True, all levels are processed simultaneously.
        :param n_head_channels: Number of head channels for multi-head operations.
        """
        super().__init__()

        self.grid_layers = grid_layers
        self.global_level_in = global_level_in
        self.global_levels = global_levels_out
        self.global_levels.sort()
        self.simultaneous = simultaneous

        # Dictionary to store the aggregation layers
        self.aggregation_layers = nn.ModuleDict()

        global_level_in_layer = global_level_in
        for k, global_level in enumerate(self.global_levels):
            if global_level_in != global_level:
                global_level = int(global_level)
                model_dim_in = model_dim_in if simultaneous or 'model_dim_out' not in locals() else model_dim_out
                model_dim_out = model_dims[k]

                # Initialize aggregation layer for the current global level
                self.aggregation_layers[str(global_level)] = NoLayer(
                    grid_layers,
                    global_level_in_layer,
                    global_level,
                    model_dim_in,
                    model_dim_out,
                    no_layer_settings,
                    n_head_channels=n_head_channels,
                    precompute_rel_coordinates=False if k==0 and first_encoder else True
                )

                # Update the input level for the next iteration
                global_level_in_layer = global_level_in if simultaneous else global_level

    def forward(self, 
                x: torch.Tensor, 
                indices_layers: dict, 
                drop_mask: torch.Tensor = None, 
                coords_in: torch.Tensor = None, 
                sample_dict: dict = None) -> tuple:
        """
        Forward pass to process the input tensor through hierarchical grid levels.

        :param x: Input tensor of shape (batch_size, num_nodes, num_features).
        :param indices_layers: Dictionary containing indices for different grid layers.
        :param drop_mask: Optional tensor mask to handle dropped elements.
        :param coords_in: Optional tensor of input coordinates.
        :param sample_dict: Dictionary containing additional sampling information.
        :return: A tuple containing:
            - x_levels: Dictionary of tensors processed at each global level.
            - drop_masks_level: Dictionary of masks applied at each global level.
        """
        x_levels = {}
        drop_masks_level = {}

        # Iterate from fine to coarse grid levels
        for k, global_level in enumerate(self.global_levels):
            global_level = int(global_level)
            global_level_in = int(self.global_level_in) if k == 0 else int(self.global_levels[k - 1])
            global_level_in = int(self.global_level_in) if self.simultaneous else global_level_in
            
            # Determine the input data and drop mask
            x_in = x if self.simultaneous or k == 0 else x_
            drop_mask_in = drop_mask if self.simultaneous or k == 0 else drop_mask_

            if str(global_level) in self.aggregation_layers.keys():
                # Process through the aggregation layer
                if coords_in is None or k > 0:
                    x_, drop_mask_ = self.aggregation_layers[str(global_level)](
                        x_in, indices_layers=indices_layers, sample_dict=sample_dict, mask=drop_mask_in
                    )
                else:
                    x_, drop_mask_ = self.aggregation_layers[str(global_level)](
                        x_in, indices_layers=indices_layers, mask=drop_mask_in, sample_dict=sample_dict, coordinates=coords_in
                    )
            else:
                x_ = x_in
                drop_mask_ = drop_mask

            # Store the processed data and drop mask
            x_levels[global_level] = x_
            drop_masks_level[global_level] = drop_mask_

        return x_levels, drop_masks_level