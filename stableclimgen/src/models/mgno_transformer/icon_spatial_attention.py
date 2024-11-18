import torch
import torch.nn as nn

from ...utils.grid_utils_icon import sequenize
from .pos_embedding import PositionEmbedder
from ..modules.transformer_modules import MultiHeadAttentionBlock
from .icon_grids import RelativeCoordinateManager

class IconSpatialAttention(nn.Module):
    """
    A module implementing spatial attention 

    Attributes:
        grid_layers (list): List of grid layers used for calculating position embeddings.
        nh_attention (bool): Whether to use neighborhood attention.
        max_seq_level (int): Maximum sequence level for attention.
        continous_pos_embedding (bool): Whether to use continuous positional embeddings. Other options are not supported
        grid_layer (nn.Module): Grid layer corresponding to the specified global level.
        global_level (int): Global level used for determining attention structure.
        coord_system (str): Coordinate system used for positional embeddings ('cartesian' or 'polar').
        position_embedder (PositionEmbedder): Positional embedding calculation module.
        embedding_layer (nn.Linear): Linear layer for generating shift and scale embeddings.
        layer_norm (nn.LayerNorm): Layer normalization module.
        MHA (nn.Module): Multi-head attention block.
        mlp_layer (nn.Sequential): Feed-forward MLP block with SiLU activation.
        gamma_mlp (nn.Parameter): Learnable parameter for scaling MLP output.
        gamma (nn.Parameter): Learnable parameter for scaling attention output.
    """

    def __init__(self,
                 grid_layers: list,
                 global_level: int,
                 model_dim: int,
                 n_head_channels: int,
                 seq_level_attention: int,
                 nh: int = 1,
                 pos_emb_calc: str = 'cartesian_km',
                 emb_table_bins: int = 16,
                 nh_attention: bool = False, 
                 continous_pos_embedding: bool = True,
                 rotate_coord_system=True) -> None:
        """
        Initializes the IconSpatialAttention module.

        :param grid_layers: List of grid layers for positional embedding calculations.
        :param global_level: The global hierarchical level for attention.
        :param model_dim: The dimensionality of the model.
        :param n_head_channels: Number of channels per attention head.
        :param seq_level_attention: Sequence level for attention operations.
        :param nh: Number of neighborhood levels, default is 1.
        :param pos_emb_calc: Method for calculating position embeddings ('cartesian_km' or 'polar').
        :param emb_table_bins: Number of bins for the embedding table, default is 16.
        :param nh_attention: Flag to indicate if neighborhood attention is used, default is False.
        :param continous_pos_embedding: Flag to use continuous positional embeddings, default is True.
        """
        super().__init__()
        
        self.grid_layers = grid_layers
        self.nh_attention = nh_attention 
        self.max_seq_level = seq_level_attention
        self.continous_pos_embedding = continous_pos_embedding
        self.grid_layer = grid_layers[global_level]
        self.global_level = global_level


        # Setup the coordinate system based on the chosen positional embedding method
        if continous_pos_embedding:
            if 'cartesian' in pos_emb_calc:
                self.coord_system = 'cartesian'
            else:
                self.coord_system = 'polar'
        
            self.position_embedder = PositionEmbedder(0, 0, emb_table_bins, model_dim, pos_emb_calc=pos_emb_calc)

        self.rel_coord_mngr = RelativeCoordinateManager(
            self.grid_layer,
            nh_input= nh_attention,
            precompute=True,
            seq_len_input=None if nh_attention else seq_level_attention,
            coord_system=self.coord_system,
            rotate_coord_system=rotate_coord_system)
        
        # Linear layer to compute shift and scale embeddings
        self.embedding_layer = nn.Linear(model_dim, model_dim * 2)
        self.layer_norm = nn.LayerNorm(model_dim, elementwise_affine=True)
        
        # Multi-head attention block configuration
        n_heads = model_dim // n_head_channels
        self.MHA = MultiHeadAttentionBlock(
            model_dim, model_dim, n_heads, input_dim=model_dim, qkv_proj=True
        )   

        # Feed-forward MLP block with normalization and activation
        self.mlp_layer = nn.Sequential(
            nn.LayerNorm(model_dim, elementwise_affine=True),
            nn.Linear(model_dim, model_dim, bias=False),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim, bias=False)
        )
    
        # Learnable scaling parameters
        self.gamma_mlp = nn.Parameter(torch.ones(model_dim) * 1e-6, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(model_dim) * 1e-6, requires_grad=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, indices_grid_layers: list, batch_dict: dict) -> tuple:
        """
        Forward pass for the IconSpatialAttention module.

        :param x: Input tensor of shape (batch, cells, vertices, features).
        :param mask: Mask tensor for attention.
        :param indices_grid_layers: List of indices for the grid layers at different levels.
        :param batch_dict: Dictionary containing additional batch-related information.
        :return: A tuple containing the processed tensor and the updated mask.
        """
        b, n, nv, f = x.shape

        # Get position embeddings using the grid layer

        rel_coords = self.rel_coord_mngr(indices_in=indices_grid_layers[int(self.global_level)])
        pos_embeddings = self.position_embedder(rel_coords[0], rel_coords[1])
       
        x = x.view(b, -1, nv, f)
        x_res = x 

        # If neighborhood attention is enabled, gather neighborhood information
        if self.nh_attention:
            x, mask = self.grid_layer.get_nh(
                x, indices_grid_layers[int(self.global_level)], batch_dict, mask=mask
            )

            if mask is not None:
                mask_update = mask.clone()
                mask_update = mask_update.sum(dim=-2) == mask_update.shape[-2]
                mask_update = mask_update.view(b, n, nv)
        else:
            # If no neighborhood attention, process sequentially
            x = sequenize(x, max_seq_level=self.max_seq_level)

            if mask is not None:
                mask = sequenize(mask, max_seq_level=self.max_seq_level)
                mask_update = mask.clone().transpose(-1, -2)
                mask_update[(mask_update.sum(dim=-1) != mask_update.shape[-1])] = False
                mask_update = mask_update.transpose(-1, -2)
                mask_update = mask_update.view(b, n, nv)
            else:
                mask_update = mask = None

        # Apply positional embeddings
        shift, scale = self.embedding_layer(pos_embeddings).transpose(-2,-3).chunk(2, dim=-1)
        x = self.layer_norm(x) * (scale + 1) + shift

        b, n_seq, nh, nv, f = x.shape
        x = x.view(b * n_seq, nh * nv, f)

        if mask is not None:
            mask = mask.view(b * n_seq, -1)

        # Set up query and key-value tensors for attention
        if self.nh_attention:
            q = x[:, [0]]
            kv = x
        else:
            q = kv = x
            
        # Apply multi-head attention
        x = self.MHA(q=q, k=kv, v=kv, mask=mask)
        x = x.view(b, n, nv, f)

        # Apply residual connections with scaling
        x = x_res + self.gamma * x
        x = x + self.gamma_mlp * self.mlp_layer(x)

        return x, mask_update