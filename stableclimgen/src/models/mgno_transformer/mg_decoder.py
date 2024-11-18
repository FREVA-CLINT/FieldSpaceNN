import torch
import torch.nn as nn

from .attention import MultiGridChannelAttention
from .neural_operator import NoLayer


class MultiGridDecoder(nn.Module):
    """
    A decoder module designed for hierarchical grid-based data processing, which aggregates
    and transforms data from multiple grid levels using projection and reduction layers.
    
    This class is typically used to decode or upsample features from different grid levels 
    into a consistent representation using attention mechanisms and various aggregation strategies.

    Attributes:
        grid_layers (dict): A dictionary containing grid layer configurations.
        global_levels_in (torch.Tensor): The input global levels as a tensor.
        global_levels (torch.Tensor): The output global levels as a tensor.
        projection_layers (nn.ModuleDict): A dictionary of projection layers for transforming
            data between different grid levels.
        multi_grid_reduction_layers (nn.ModuleDict): A dictionary of reduction layers that 
            aggregate data from multiple input levels into a single output level.
    """

    def __init__(self, 
                 grid_layers: dict, 
                 global_levels_in: torch.Tensor, 
                 global_levels: list, 
                 model_dims_in: list, 
                 model_dims_out: list,
                 no_layer_settings, 
                 n_head_channels: int = 16,
                 output_dim: int=None) -> None:
        """
        Initializes the MultiGridDecoder for hierarchical feature aggregation.

        :param grid_layers: A dictionary of grid layers for feature transformation.
        :param global_levels_in: A tensor specifying the initial global levels.
        :param global_levels: A list of global levels to process data through.
        :param model_dims_in: A list specifying the input dimensionality for each grid level.
        :param model_dims_out: A list specifying the output dimensionality for each grid level.
        :param no_layer_settings: Configuration settings for the `no_layer`.
        :param n_head_channels: Number of channels per head for multi-head operations.
        """
        super().__init__()

        self.grid_layers = grid_layers
        self.global_levels_in = global_levels_in = torch.tensor(global_levels_in)
        self.global_levels = global_levels = torch.tensor(global_levels)
        model_dims_in = torch.tensor(model_dims_in)
        model_dims_out = torch.tensor(model_dims_out)

        # Dictionaries to store projection and reduction layers
        self.projection_layers = nn.ModuleDict()
        self.multi_grid_reduction_layers = nn.ModuleDict()

        for k, global_level_output in enumerate(global_levels):
            # Filter the input levels greater than or equal to the current output level
            global_levels_in_step = global_levels_in[global_levels_in >= global_level_output]
            model_dims_in_step = model_dims_in[global_levels_in >= global_level_output]

            self.projection_layers_step = nn.ModuleDict()

            # Create projection layers for each input level
            for j, global_level_in_step in enumerate(global_levels_in_step):
                if global_level_in_step > global_level_output:
                    self.projection_layers_step[str(int(global_level_in_step))] = NoLayer(
                        grid_layers,
                        int(global_level_in_step),
                        int(global_level_output),
                        int(model_dims_in_step[j]), 
                        int(model_dims_in_step[j]),
                        no_layer_settings,
                        n_head_channels=n_head_channels
                    )
                else:
                    self.projection_layers_step[str(int(global_level_in_step))] = nn.Identity()

            # Update the input model dimensions and global levels
            model_dims_in = torch.concat(
                (model_dims_out[k].view(-1), model_dims_in[global_levels_in < global_level_output])
            )
            global_levels_in = torch.concat(
                (global_level_output.view(-1), global_levels_in[global_levels_in < global_level_output])
            )

            # Store the projection and reduction layers
            self.projection_layers[str(int(global_level_output))] = self.projection_layers_step
            self.multi_grid_reduction_layers[str(int(global_level_output))] = MultiGridChannelAttention(
                model_dims_in_step,
                int(model_dims_out[k]),
                n_head_channels=n_head_channels
            )
        
        if output_dim is not None:
            self.mlp_layer_out = nn.Sequential(
                nn.Linear(int(model_dims_out[k]), int(model_dims_out[k]) // 2, bias=False),
                nn.SiLU(),
                nn.Linear(int(model_dims_out[k]) // 2, output_dim, bias=False)
            )
        else:
            self.mlp_layer_out = nn.Identity()

    def forward(self, 
                x_levels: dict, 
                indices_grid_layers: dict, 
                drop_mask_levels: dict = None, 
                sample_dict: dict = None, 
                coords_out: torch.Tensor = None) -> tuple:
        """
        Forward pass to aggregate and transform input data across multiple grid levels.

        :param x_levels: A dictionary containing tensors for each input grid level.
        :param indices_grid_layers: A dictionary with indices for the grid layers.
        :param drop_mask_levels: Optional dictionary of drop masks for each grid level.
        :param sample_dict: Optional dictionary for sampling-related information.
        :param coords_out: Optional tensor of output coordinates.
        :return: A tuple containing:
            - x_levels[int(global_level_output)]: The final output tensor at the highest grid level.
            - drop_mask_levels[int(global_level_output)]: The final drop mask for the highest grid level.
        """
        # Iterate over the output global levels and aggregate data
        for global_level_output, projection_layers_output in self.projection_layers.items():
            x_out = []

            # Collect and project data from each input level
            for global_level_input, projection_layers_input in projection_layers_output.items():
                if drop_mask_levels is not None:
                    drop_mask_input = drop_mask_levels.get(int(global_level_input), None)

                if int(global_level_input) > int(global_level_output):
                    x, drop_mask_level = projection_layers_input(
                        x_levels[int(global_level_input)], 
                        indices_layers=indices_grid_layers,
                        sample_dict=sample_dict,
                        mask=drop_mask_input
                    )
                else:
                    x = x_levels[int(global_level_input)]
                    drop_mask_level = drop_mask_input

                x_out.append(x)

                if drop_mask_level is not None:
                    mask_shape = x.shape[:-1]

                    if int(global_level_output) in drop_mask_levels and drop_mask_input is not None:
                        drop_mask_levels[int(global_level_output)] = torch.logical_and(
                            drop_mask_levels[int(global_level_output)].view(mask_shape), 
                            drop_mask_level.view(mask_shape)
                        )
                    else:
                        drop_mask_levels[int(global_level_output)] = drop_mask_level.view(mask_shape)

            # Aggregate data using the multi-grid reduction layer
            x_levels[int(global_level_output)] = self.multi_grid_reduction_layers[global_level_output](x_out)

        x = self.mlp_layer_out(x_levels[int(global_level_output)])
        return x, drop_mask_levels[int(global_level_output)]