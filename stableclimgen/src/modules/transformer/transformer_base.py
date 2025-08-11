from itertools import zip_longest
from typing import Union, List, Optional, Dict

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.functional import scaled_dot_product_attention

from ...modules.grids.grid_layer import GridLayer
from ...modules.embedding.embedder import EmbedderSequential
from stableclimgen.src.modules.rearrange import (RearrangeTimeCentric, RearrangeSpaceCentric,
                                                 RearrangeVarCentric, RearrangeNHCentric, RearrangeVarNHCentric)
from stableclimgen.src.utils.utils import EmbedBlock

from ...utils.helpers import check_value

from ..base import get_layer, LinEmbLayer, IdentityLayer, MLP_fac


def safe_scaled_dot_product_attention(q, k, v, mask=None, is_causal=False, chunk_size=2**16):
    """
    Applies scaled dot-product attention with batch chunking to avoid CUDA issues.

    Args:
        q: (B, H, L_q, D)
        k: (B, H, L_k, D)
        v: (B, H, L_k, D)
        mask: Optional (B, H, L_q, L_k)
        is_causal: Boolean
        chunk_size: Chunk size for batch splitting

    Returns:
        Tensor of shape (B, H, L_q, D)
    """
    B = q.shape[0]
    chunks = [
        scaled_dot_product_attention(
            q[i:i+chunk_size],
            k[i:i+chunk_size],
            v[i:i+chunk_size],
            attn_mask=None if mask is None else mask[i:i+chunk_size],
            is_causal=is_causal
        )
        for i in range(0, B, chunk_size)
    ]
    return torch.cat(chunks, dim=0)


class SelfAttention(nn.Module):
    """
    Self-attention layer with optional embeddings and causal mask support.

    This module implements the scaled dot-product attention mechanism with optional
    causal masking, suitable for time-series or sequence data.

    :param in_features: Number of input channels.
    :param out_features_list: Number of output channels.
    :param num_heads: Number of attention heads.
    :param dropout: Dropout probability for the output. Default is 0.
    :param is_causal: Whether to apply causal masking (True/False). Default is False.
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            num_heads: int,
            dropout: float = 0.,
            is_causal: bool = False,
            layer_confs = {},
            qkv_proj = False,
            cross = False
    ):
        super().__init__()


        if qkv_proj:
            self.q_proj = nn.Linear(in_features, in_features, bias=False)
            self.kv_proj = nn.Linear(in_features, in_features, bias=False)
            self.out_layer = nn.Linear(in_features, out_features, bias=True) if in_features!=out_features else nn.Identity()
        else:
            self.q_proj = nn.Identity()
            self.kv_proj = nn.Identity()
            self.out_layer = nn.Identity()

        self.proj_fcn = self.proj_xkv if cross else self.proj_x

        # Learnable scaling parameter to control the output's magnitude
        self.gamma = torch.nn.Parameter(torch.ones(out_features) * 1E-6)

        self.n_heads = num_heads  # Number of attention heads
        self.dropout = dropout  # Dropout probability for attention output

        self.is_causal = is_causal  # Flag for causal masking (used in time-series tasks)

    def proj_x(self, x, kv: torch.Tensor=None):
        return self.q_proj(x).chunk(3,dim=-1)
    
    def proj_xkv(self, x, kv: torch.Tensor):
        return self.q_proj(x), *self.kv_proj(kv).chunk(2,dim=-1)

    def forward(self, x: torch.Tensor, kv: torch.Tensor=None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the SelfAttention layer.

        This function computes the attention mechanism using query, key, and value projections
        followed by applying the scaled dot-product attention with optional masking.

        :param x: Input tensor of shape (batch, time, channels).
        :param mask: Optional mask tensor (used for causal or attention masking).
        :return: Output tensor after applying self-attention and projection.
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


    def scaled_dot_product_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                     mask: Optional[torch.Tensor], **kwargs) -> torch.Tensor:
        """
        Scaled dot-product attention mechanism, with optional causal masking.

        :param q: Query tensor of shape [batch, n_heads, time, d_head].
        :param k: Key tensor of shape [batch, n_heads, time, d_head].
        :param v: Value tensor of shape [batch, n_heads, time, d_head].
        :param mask: Optional mask tensor for attention.
        :return: Output tensor after applying attention mechanism.
        """
        if mask is not None:
            mask = mask==False if mask.dtype==torch.bool else 1-mask
        return safe_scaled_dot_product_attention(q, k, v, mask=mask, is_causal=self.is_causal)


class NHAttention(nn.Module):

    def __init__(
            self,
            grid_layer: GridLayer,
            in_features: int,
            out_features: int,
            num_heads: int,
            dropout: float = 0.,
            is_causal: bool = False,
            layer_confs = {},
            with_variable_attention = False
    ):
        super().__init__()

        self.attention = SelfAttention(
            in_features, 
            out_features, 
            num_heads,
            layer_confs=layer_confs,
            dropout=dropout,
            is_causal=is_causal,
            qkv_proj=True,
            cross=True
            )
        
        self.grid_layer = grid_layer

        if with_variable_attention:
            self.nh_pattern = 'b t s nh v c -> (b t s) (nh v) c'
            self.pattern = 'b t s v c -> (b t s) v c'
            self.reverse_pattern = '(b t s) v c -> b t s v c'
        else:
            self.nh_pattern = 'b t s nh v c -> (b t s v) (nh) c'
            self.pattern = 'b t s v c -> (b t s v) 1 c'
            self.reverse_pattern = '(b t s v) 1 c -> b t s v c'


    def forward(self, x: torch.Tensor, emb, mask=None, sample_configs: Dict={}) -> torch.Tensor:
        b, t, s, v, c = x.shape

        x_nh, mask_nh = self.grid_layer.get_nh(x, **sample_configs, with_nh=True, mask=mask)

        x = rearrange(x, self.pattern)
        x_nh = rearrange(x_nh, self.nh_pattern)
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
            layer_confs = {}
    ):
        super().__init__()

        # Define MLP with a hidden layer, GELU activation, and optional dropout
        self.branch_layer1 = get_layer(in_features, in_features * mult, layer_confs=layer_confs, bias=True)
        self.branch_layer2 = get_layer(in_features * mult, out_features_list, layer_confs=layer_confs, bias=True)

        self.activation = torch.nn.GELU()
        self.dropout = nn.Dropout(p=dropout) if dropout>0 else nn.Identity()

        # Learnable scaling parameter for the output of the MLP layer
        self.gamma = torch.nn.Parameter(torch.ones(out_features_list) * 1E-6)

    def forward(self, x: torch.Tensor, emb: Dict) -> torch.Tensor:
        """
        Forward pass for the MLPLayer.

        This function applies the MLP transformations and adds the skip connection to the output.

        :param x: Input tensor.
        :return: Output tensor after MLP transformation and skip connection.
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
            seq_lengths: List[int] = None,
            num_heads: List[int] = 1,
            n_head_channels: List[int] = None,
            att_dims: List[int] = None,
            mlp_mult: List[int] = 1,
            dropout: List[float] = 0.,
            spatial_dim_count: int = 1,
            embedders: List[EmbedderSequential] = None,
            layer_confs: Dict = {},
            layer_confs_emb={},
            **kwargs
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
                #qkv_proj = len(layer_confs) == 0
                q_layer = get_layer(att_dim, [att_dim], layer_confs=layer_confs) 
                kv_layer = get_layer(att_dim, [2, att_dim], layer_confs=layer_confs, bias=True) 
                out_layer = get_layer(att_dim, out_features_list[i], layer_confs=layer_confs, bias=True) if att_dim != out_features_list[i] else IdentityLayer()

                cross = False
                # Select rearrangement function based on block type
                if block == "t":
                    rearrange_fn = RearrangeTimeCentric
                elif block == "s":
                    seq_length = 4**seq_length
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

                trans_block = rearrange_fn(SelfAttention(att_dim, att_dim, n_heads, layer_confs=layer_confs, qkv_proj=False, cross=cross), spatial_dim_count, seq_length, proj_layer_q=q_layer, proj_layer_kv=kv_layer, out_layer=out_layer, grid_layer=kwargs['grid_layer'])
     
            lin_emb_layers.append(LinEmbLayer(in_features, att_dim, layer_confs=layer_confs, identity_if_equal=True, embedder=embedders[i], layer_norm=True, layer_confs_emb=layer_confs_emb))

            # Skip connection layer: Identity if in_features == out_features_list, else a linear projection
            if in_features != out_features_list[i]:
                residual = get_layer(in_features, out_features_list[i], layer_confs=layer_confs, bias=False)
            else:
                residual = IdentityLayer()

            # append normalization and trans_block
            trans_blocks.append(trans_block)
            residuals.append(residual)

            in_features = out_features_list if isinstance(out_features_list, int) else out_features_list[i]

        self.lin_emb_layers = nn.ModuleList(lin_emb_layers)
        self.blocks = nn.ModuleList(trans_blocks)
        self.residuals = nn.ModuleList(residuals)

    def forward(self, x: torch.Tensor, kv: torch.Tensor=None, emb: Optional[Dict] = None, mask: Optional[torch.Tensor] = None,
                sample_configs: Optional[Dict] = None, *args) -> torch.Tensor:
        """
        Forward pass for the TransformerBlock.

        This function applies each block (MLP, Self-Attention) sequentially, optionally
        embedding the input at each stage and applying normalization.

        :param x: Input tensor.
        :param emb: Optional embedding tensor to modify input at each block.
        :param mask: Optional mask tensor (e.g., for causal masking).
        :param cond: Optional conditioning tensor (additional input).
        :return: Output tensor after applying all blocks sequentially.
        """
        for block, lin_emb_layer, residual in zip(self.blocks, self.lin_emb_layers, self.residuals):

            out = lin_emb_layer(x, emb=emb, sample_configs=sample_configs)
            x = block(out, emb=emb, sample_configs=sample_configs, mask=mask) + residual(x, emb=emb)

        return x


