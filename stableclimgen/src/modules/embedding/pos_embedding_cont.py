import torch
import torch.nn as nn
from ...modules.transformer import transformer_modules as helpers

class PositionEmbedder(nn.Module):
    """
    A module to embed positional information for various coordinate systems (polar, cartesian) with
    different embedding strategies (e.g., learned, discrete, log, etc.).

    Attributes:
        pos_emb_calc (str): The method used for position embedding calculation (e.g., 'cartesian', 'polar').
        operation (str or None): The operation to apply on the embeddings ('sum' or 'product').
        transform (function or None): Transformation function for positional values (e.g., logarithmic transformations).
        proj_layer (nn.Sequential or None): Projection layer to map position embeddings to the correct dimensionality.
        cartesian (bool): Flag indicating if Cartesian embedding is used.
        km_transform (bool): Flag indicating if km transformation should be applied.
        pos1_emb (nn.Module or function): Position embedding module for the first positional input.
        pos2_emb (nn.Module or function): Position embedding module for the second positional input (e.g., phi values).
    """

    def __init__(self, 
                 min_dist: float, 
                 max_dist: float, 
                 emb_table_bins: int, 
                 emb_dim: int, 
                 pos_emb_calc: str = "polar", 
                 phi_table: torch.Tensor = None) -> None:
        """
        Initializes the PositionEmbedder class for different positional embedding strategies.

        :param min_dist: Minimum distance value for positional embedding calculation.
        :param max_dist: Maximum distance value for positional embedding calculation.
        :param emb_table_bins: Number of bins for the embedding table.
        :param emb_dim: The dimensionality of the embeddings.
        :param pos_emb_calc: Strategy for positional embedding calculation ('polar', 'cartesian', 'learned', etc.).
        :param phi_table: Predefined phi values for embedding (optional). If not provided, a default embedding will be used.
        """
        super().__init__()
        
        self.pos_emb_calc = pos_emb_calc
        self.operation = None
        self.transform = None
        self.proj_layer = None
        self.cartesian = False

        # Initialize based on position embedding calculation method
        if "descrete" in pos_emb_calc and "polar" in pos_emb_calc:
            self.pos1_emb = helpers.PositionEmbedder_phys_log(min_dist, max_dist, emb_table_bins, n_heads=emb_dim)
            self.pos2_emb = phi_table if phi_table is not None else helpers.PositionEmbedder_phys(-torch.pi, torch.pi, emb_table_bins, n_heads=emb_dim, special_token=True)

        if "semi" in pos_emb_calc and "polar" in pos_emb_calc:
            self.pos1_emb = nn.Sequential(nn.Linear(1, emb_dim), nn.SiLU())
            self.pos2_emb = phi_table if phi_table is not None else helpers.PositionEmbedder_phys(-torch.pi, torch.pi, emb_table_bins, n_heads=emb_dim, special_token=True)

        if "cartesian" in pos_emb_calc:
            self.proj_layer = nn.Sequential(
                nn.Linear(2, emb_dim, bias=True),
                nn.SiLU(),
                nn.Linear(emb_dim, emb_dim, bias=False),
                nn.Sigmoid()
            )
            self.cartesian = True

        if "learned" in pos_emb_calc and "polar" in pos_emb_calc:
            self.proj_layer = nn.Sequential(
                nn.Linear(2 * emb_dim, emb_dim, bias=True),
                nn.SiLU(),
                nn.Linear(emb_dim, emb_dim, bias=False),
                nn.Sigmoid()
            )

        # km transformation flag
        self.km_transform = 'km' in pos_emb_calc

        # Transform function based on the selected embedding strategy
        if 'inverse' in pos_emb_calc:
            self.transform = helpers.conv_coordinates_inv
        elif 'sig_log' in pos_emb_calc:
            self.transform = helpers.conv_coordinates_sig_log
        elif 'sig_inv_log' in pos_emb_calc:
            self.transform = helpers.conv_coordinates_inv_sig_log  
        elif 'log' in pos_emb_calc:
            self.transform = helpers.conv_coordinates_log
        
        # Define operation (sum or product) on the embeddings
        if 'sum' in pos_emb_calc:
            self.operation = 'sum'
        elif 'product' in pos_emb_calc:
            self.operation = 'product'

    def forward(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        """
        Computes the position embeddings based on the chosen coordinate system and embedding strategy.

        :param pos1: Tensor containing the first set of positional values (e.g., radial distance).
        :param pos2: Tensor containing the second set of positional values (e.g., angular values).
        :return: The computed position embeddings tensor.
        """
        if self.cartesian:
            # Apply km transformation if required
            if self.km_transform:
                pos1 = pos1 * 6371.
                pos2 = pos2 * 6371.
                pos1[pos1.abs() < 0.01] = 0
                pos2[pos2.abs() < 0.01] = 0
            else:
                pos1[pos1.abs() < 1e-6] = 0
                pos2[pos2.abs() < 1e-6] = 0

            # Apply transform function if defined
            if self.transform is not None:
                pos1 = self.transform(pos1)
                pos2 = self.transform(pos2)
            
            # Return the projected embedding
            return 16 * self.proj_layer(torch.stack((pos1, pos2), dim=-1))    

        else:
            # Apply km transformation if required
            if self.km_transform:
                pos1 = pos1 * 6371.
                dist_0 = pos1 < 0.01
            else:
                dist_0 = pos1 < 1e-6

            # Apply transformation if needed
            if self.transform is not None:
                pos1 = self.transform(pos1)
            
            if isinstance(self.pos1_emb, nn.Sequential):
                pos1 = pos1.unsqueeze(dim=-1)

            pos1_emb = self.pos1_emb(pos1)
            pos2_emb = self.pos2_emb(pos2, special_token_mask=dist_0)

            # Return concatenated or operated embeddings
            if self.proj_layer is not None:
                return 16 * self.proj_layer(torch.concat((pos1_emb, pos2_emb), dim=-1))
                        
            if self.operation == 'sum':
                return pos1_emb + pos2_emb
            elif self.operation == 'product':
                return pos1_emb * pos2_emb