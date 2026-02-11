from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.functional import scaled_dot_product_attention

from ...modules.grids.grid_layer import GridLayer
from ...modules.embedding.embedder import EmbedderSequential
from ..rearrange import (RearrangeTimeCentric, RearrangeSpaceCentric,
                         RearrangeVarCentric, RearrangeNHCentric, RearrangeVarNHCentric)
from ..cnn.cnn_base import EmbedBlock

from ...utils.helpers import check_value

from ..base import get_layer, IdentityLayer, MLP_fac
from ..field_space.field_space_base import LinEmbLayer


def safe_scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    chunk_size: int = 2**16
):
    """
    Apply scaled dot-product attention with batch chunking to avoid CUDA issues.

    :param q: Query tensor of shape ``(b, h, l_q, d)``.
    :param k: Key tensor of shape ``(b, h, l_k, d)``.
    :param v: Value tensor of shape ``(b, h, l_k, d)``.
    :param mask: Optional attention mask of shape ``(b, h, l_q, l_k)``.
    :param is_causal: Whether to apply causal masking.
    :param chunk_size: Chunk size for batch splitting.
    :return: Attention output of shape ``(b, h, l_q, d)``.
    """
    B, H, _, _ = q.shape

    # Reduce chunk size per head to stay within kernel limits.
    safe_chunk_size = chunk_size // H

    if mask is not None:
        mask = mask==False if mask.dtype==torch.bool else mask

    if B <= safe_chunk_size:
        return scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=is_causal
        )

    results = []
    for i in range(0, B, safe_chunk_size):
        q_chunk = q[i:i + safe_chunk_size]
        k_chunk = k[i:i + safe_chunk_size]
        v_chunk = v[i:i + safe_chunk_size]

        mask_chunk = None
        if mask is not None:
            if mask.dim() == 4:
                mask_chunk = mask[i:i + safe_chunk_size]
            else:
                mask_chunk = mask
        
        chunk_result = scaled_dot_product_attention(
            q_chunk,
            k_chunk,
            v_chunk,
            attn_mask=mask_chunk,
            dropout_p=0.0,
            is_causal=is_causal,
        )
        results.append(chunk_result)

    return torch.cat(results, dim=0)


class SelfAttention(nn.Module):
    """
    Self-attention layer with optional embeddings and causal mask support.

    This module implements the scaled dot-product attention mechanism with optional
    causal masking, suitable for time-series or sequence data.

    :param in_features: Number of input channels.
    :param out_features: Number of output channels.
    :param num_heads: Number of attention heads.
    :param dropout: Dropout probability for the output. Default is 0.
    :param is_causal: Whether to apply causal masking. Default is False.
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            num_heads: int,
            dropout: float = 0.,
            is_causal: bool = False,
            layer_confs: Dict[str, Any] = {},
            qkv_proj: bool = False,
            cross: bool = False
    ):
        super().__init__()


        if qkv_proj:
            self.q_proj: nn.Module = nn.Linear(in_features, in_features, bias=False)
            self.kv_proj: nn.Module = nn.Linear(in_features, in_features, bias=False)
        else:
            self.q_proj: nn.Module = nn.Identity()
            self.kv_proj: nn.Module = nn.Identity()
        
        self.out_layer: nn.Module = nn.Linear(in_features, out_features, bias=True) if in_features!=out_features else nn.Identity()

        self.proj_fcn: Callable[..., Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = self.proj_xkv if cross else self.proj_x

        # Learnable scaling parameter to control the output's magnitude
        self.gamma: torch.nn.Parameter = torch.nn.Parameter(torch.ones(out_features) * 1E-6)

        self.n_heads: int = num_heads  # Number of attention heads
        self.dropout: float = dropout  # Dropout probability for attention output

        self.is_causal: bool = is_causal  # Flag for causal masking (used in time-series tasks)

    def proj_x(self, x: torch.Tensor, kv: Optional[torch.Tensor] = None):
        """
        Project a single input into query, key, and value tensors.

        :param x: Input tensor of shape ``(b, l, f)``.
        :param kv: Optional key/value tensor (unused).
        :return: Tuple of (q, k, v) tensors of shape ``(b, l, f/3)``.
        """
        return self.q_proj(x).chunk(3,dim=-1)
    
    def proj_xkv(self, x: torch.Tensor, kv: torch.Tensor):
        """
        Project separate query and key/value inputs.

        :param x: Query input tensor of shape ``(b, l_q, f)``.
        :param kv: Key/value input tensor of shape ``(b, l_k, f)``.
        :return: Tuple of (q, k, v) tensors.
        """
        return self.q_proj(x), *self.kv_proj(kv).chunk(2,dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        kv: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        **kwargs: Any
    ):
        """
        Forward pass for the SelfAttention layer.

        This function computes the attention mechanism using query, key, and value projections
        followed by applying the scaled dot-product attention with optional masking.

        :param x: Input tensor of shape ``(b, l_q, f)``.
        :param kv: Optional key/value tensor of shape ``(b, l_k, f)``.
        :param mask: Optional attention mask of shape ``(b, h, l_q, l_k)``.
        :param kwargs: Additional keyword arguments (unused).
        :return: Output tensor of shape ``(b, l_q, f_out)``.
        """
        q, k, v = self.proj_fcn(x, kv=kv)

        b, t_g_v, c = x.shape
         
        # Rearrange tensors for multi-head attention: [batch, time, (n_heads * d_head)] -> [batch, n_heads, time, d_head]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.n_heads), (q, k, v))

        # Apply scaled dot-product attention
        attn_out = self.scaled_dot_product_attention(q, k, v, mask)

        # Reshape and project output from multi-head attention back to the original dimensions
        attn_out = rearrange(attn_out, "b h t_g_v c -> b t_g_v (h c)", b=b, t_g_v=t_g_v, h=self.n_heads)

        # Apply skip connection and scaling to the output
        return self.gamma * self.out_layer(attn_out)


    def scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
        **kwargs: Any
    ):
        """
        Scaled dot-product attention mechanism, with optional causal masking.

        :param q: Query tensor of shape ``(b, h, l_q, d)``.
        :param k: Key tensor of shape ``(b, h, l_k, d)``.
        :param v: Value tensor of shape ``(b, h, l_k, d)``.
        :param mask: Optional attention mask of shape ``(b, h, l_q, l_k)``.
        :param kwargs: Additional keyword arguments (unused).
        :return: Output tensor after applying attention, shape ``(b, h, l_q, d)``.
        """
        return safe_scaled_dot_product_attention(q, k, v, mask=mask, is_causal=self.is_causal)


class NHAttention(nn.Module):
    """
    Neighborhood attention that attends to spatial neighbors from a GridLayer.

    :param grid_layer: Grid layer providing neighborhood indices.
    :param in_features: Number of input channels.
    :param out_features: Number of output channels.
    :param num_heads: Number of attention heads.
    :param dropout: Dropout probability.
    :param is_causal: Whether to apply causal masking.
    :param layer_confs: Optional layer configuration dictionary.
    :param with_variable_attention: Whether to attend over variables within neighborhoods.
    """

    def __init__(
            self,
            grid_layer: GridLayer,
            in_features: int,
            out_features: int,
            num_heads: int,
            dropout: float = 0.,
            is_causal: bool = False,
            layer_confs: Dict[str, Any] = {},
            with_variable_attention: bool = False
    ):
        super().__init__()

        self.attention: SelfAttention = SelfAttention(
            in_features, 
            out_features, 
            num_heads,
            layer_confs=layer_confs,
            dropout=dropout,
            is_causal=is_causal,
            qkv_proj=True,
            cross=True
            )
        
        self.grid_layer: GridLayer = grid_layer

        self.nh_pattern: str
        self.pattern: str
        self.reverse_pattern: str
        if with_variable_attention:
            self.nh_pattern = 'b t s nh v c -> (b t s) (nh v) c'
            self.pattern = 'b t s v c -> (b t s) v c'
            self.reverse_pattern = '(b t s) v c -> b t s v c'
        else:
            self.nh_pattern = 'b t s nh v c -> (b t s v) (nh) c'
            self.pattern = 'b t s v c -> (b t s v) 1 c'
            self.reverse_pattern = '(b t s v) 1 c -> b t s v c'


    def forward(
        self,
        x: torch.Tensor,
        emb: Optional[Dict[str, Any]] = None,
        mask: Optional[torch.Tensor] = None,
        sample_configs: Dict[str, Any] = {}
    ):
        """
        Compute neighborhood attention over spatial neighbors.

        :param x: Input tensor of shape ``(b, t, n, v, f)``.
        :param emb: Optional embedding dictionary.
        :param mask: Optional mask tensor of shape ``(b, t, n, v, m)``.
        :param sample_configs: Sampling configuration for neighborhood selection.
        :return: Output tensor of shape ``(b, t, n, v, f_out)``.
        """
        b, t, s, v, c = x.shape

        # Gather neighborhoods and corresponding masks from the grid layer.
        x_nh, mask_nh = self.grid_layer.get_nh(x, **sample_configs, with_nh=True, mask=mask)

        x = rearrange(x, self.pattern)
        x_nh = rearrange(x_nh, self.nh_pattern)
        # Convert mask to attention-compatible form.
        mask_nh = rearrange(mask_nh == False, self.nh_pattern)
        
        x = self.attention(x, x_nh, emb=emb, mask=mask_nh)

        x = rearrange(x, self.reverse_pattern, b=b, t=t, s=s, v=v, c=c)

        return x


class MLPLayer(nn.Module):
    """
    Multi-Layer Perceptron (MLP) layer with optional embedding and dropout.

    This MLP can be used in transformer blocks for nonlinear transformations with optional
    embedding layer and dropout regularization.

    :param in_features: Number of input channels.
    :param out_features_list: Number of output channels.
    :param mult: Multiplier for the hidden channels. Default is 1.
    :param dropout: Dropout probability for the hidden layers. Default is 0.
    """

    def __init__(
            self,
            in_features: int,
            out_features_list: int,
            mult: int = 1,
            dropout: float = 0.,
            layer_confs: Dict[str, Any] = {}
    ):
        super().__init__()

        # Define MLP with a hidden layer, GELU activation, and optional dropout
        self.branch_layer1: nn.Module = get_layer(in_features, in_features * mult, layer_confs=layer_confs, bias=True)
        self.branch_layer2: nn.Module = get_layer(in_features * mult, out_features_list, layer_confs=layer_confs, bias=True)

        self.activation: nn.Module = torch.nn.GELU()
        self.dropout: nn.Module = nn.Dropout(p=dropout) if dropout>0 else nn.Identity()

        # Learnable scaling parameter for the output of the MLP layer
        self.gamma: torch.nn.Parameter = torch.nn.Parameter(torch.ones(out_features_list) * 1E-6)

    def forward(self, x: torch.Tensor, emb: Dict[str, Any]):
        """
        Forward pass for the MLPLayer.

        This function applies the MLP transformations and adds the skip connection to the output.

        :param x: Input tensor of shape ``(b, l, f)``.
        :param emb: Embedding dictionary forwarded to linear layers.
        :return: Output tensor of shape ``(b, l, f_out)``.
        """
        x = self.branch_layer1(x, emb=emb)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.branch_layer2(x, emb=emb)

        return self.gamma * x


class TransformerBlock(EmbedBlock):
    """
    Transformer block that combines MLP layers and Self-Attention layers with optional embeddings.

    This block applies a sequence of layers, such as attention or MLP, followed by normalization
    and optional embeddings (e.g., time, space, or variable embeddings) for each block.

    :param in_features: Number of input channels.
    :param out_features_list: List of output channels for each block in the sequence.
    :param blocks: List of blocks to include, can be "mlp", "t", "s", or "v" for self-attention.
    :param embed_confs: List of dictionaries containing the arguments for the embedders for each block.
    :param embed_mode: Mode for embedding application, default is "sum".
    :param num_heads: List of number of attention heads for each attention block.
    :param mlp_mult: List of multipliers for hidden channels in MLP blocks.
    :param dropout: List of dropout probabilities for each block.
    :param spatial_dim_count: Number of spatial dimensions for the input.
    """

    def __init__(
            self,
            in_features: int,
            out_features_list: List[int],
            blocks: List[str],
            seq_lengths: Optional[Union[int, List[int]]] = None,
            num_heads: Optional[Union[int, List[int]]] = None,
            n_head_channels: Union[int, List[int]] = 16,
            att_dims: Optional[Union[int, List[int]]] = None,
            mlp_mult: Union[int, List[int]] = 1,
            dropout: Union[float, List[float]] = 0.,
            spatial_dim_count: int = 1,
            embedders: Optional[List[EmbedderSequential]] = None,
            layer_confs: Dict[str, Any] = {},
            layer_confs_emb: Dict[str, Any] = {},
            **kwargs: Any
    ):
        super().__init__()

        if not out_features_list:
            out_features_list = in_features  # Default output channels to input channels if not provided

        out_features_list = check_value(out_features_list, len(blocks))
        num_heads = check_value(num_heads, len(blocks))
        n_head_channels = check_value(n_head_channels, len(blocks))
        mlp_mult = check_value(mlp_mult, len(blocks))
        dropout = check_value(dropout, len(blocks))
        seq_lengths = check_value(seq_lengths, len(blocks))
        att_dims = check_value(att_dims, len(blocks))

        embedders = check_value(embedders, len(blocks))

        trans_blocks, lin_emb_layers, norms, residuals = [], [], [], []
        for i, block in enumerate(blocks):
            
            att_dim = att_dims[i] if att_dims[i] is not None else in_features
            n_heads = num_heads[i] if not n_head_channels[i] else att_dim // n_head_channels[i]

            seq_length = seq_lengths[i]

            if block == "mlp":
                trans_block = MLP_fac(att_dim, out_features_list[i], mlp_mult[i], dropout[i], layer_confs=layer_confs, gamma=True)

            else:
                q_layer = get_layer(att_dim, [att_dim], layer_confs=layer_confs) 
                kv_layer = get_layer([1, att_dim], [2, att_dim], layer_confs=layer_confs, bias=True)
                cross = False
                # Select rearrangement function based on block type.
                if block == "t":
                    rearrange_fn = RearrangeTimeCentric
                elif block == "s":
                    seq_length = 4**seq_length if seq_length else seq_length
                    rearrange_fn = RearrangeSpaceCentric
                elif block == "snh":
                    seq_length = 1
                    cross = True
                    rearrange_fn = RearrangeNHCentric
                elif block == "vsnh":
                    seq_length = 1
                    cross = True
                    rearrange_fn = RearrangeVarNHCentric
                else:
                    assert block == "v"
                    seq_length = None
                    rearrange_fn = RearrangeVarCentric

                trans_block = rearrange_fn(SelfAttention(att_dim, out_features_list[i], n_heads, layer_confs=layer_confs, qkv_proj=False, cross=cross), spatial_dim_count, seq_length, proj_layer_q=q_layer, proj_layer_kv=kv_layer, grid_layer=kwargs['grid_layer'] if 'grid_layer' in kwargs.keys() else None)
     
            lin_emb_layers.append(LinEmbLayer(in_features, att_dim, layer_confs=layer_confs, identity_if_equal=True, embedder=embedders[i], layer_norm=True, layer_confs_emb=layer_confs_emb, spatial_dim_count=spatial_dim_count))


            # Skip connection layer: Identity if in_features == out_features_list, else a linear projection
            if in_features != out_features_list[i]:
                residual = get_layer(in_features, out_features_list[i], layer_confs=layer_confs, bias=False)
            else:
                residual = IdentityLayer()

            # append normalization and trans_block
            trans_blocks.append(trans_block)
            residuals.append(residual)

            in_features = out_features_list if isinstance(out_features_list, int) else out_features_list[i]

        self.lin_emb_layers: nn.ModuleList = nn.ModuleList(lin_emb_layers)
        self.blocks: nn.ModuleList = nn.ModuleList(trans_blocks)
        self.residuals: nn.ModuleList = nn.ModuleList(residuals)

    def forward(
        self,
        x: torch.Tensor,
        kv: Optional[torch.Tensor] = None,
        emb: Optional[Dict[str, Any]] = None,
        mask: Optional[torch.Tensor] = None,
        sample_configs: Optional[Dict[str, Any]] = None,
        *args: Any
    ):
        """
        Forward pass for the TransformerBlock.

        This function applies each block (MLP, Self-Attention) sequentially, optionally
        embedding the input at each stage and applying normalization.

        :param x: Input tensor of shape ``(b, v, t, n, d, f)`` or flattened to ``(b, l, f)``.
        :param kv: Optional key/value tensor of shape ``(b, l_k, f)``.
        :param emb: Optional embedding dictionary to modify input at each block.
        :param mask: Optional attention mask of shape ``(b, h, l_q, l_k)``.
        :param sample_configs: Optional sampling configuration for spatial blocks.
        :param args: Additional positional arguments (unused).
        :return: Output tensor with the same leading shape as x and updated feature dim.
        """
        for block, lin_emb_layer, residual in zip(self.blocks, self.lin_emb_layers, self.residuals):

            out = lin_emb_layer(x, emb=emb, sample_configs=sample_configs)
            x = block(out, emb=emb, sample_configs=sample_configs, mask=mask) + residual(x, emb=emb)

        return x
