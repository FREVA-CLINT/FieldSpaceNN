from itertools import zip_longest
from typing import Union, List, Optional, Dict

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.functional import scaled_dot_product_attention

from stableclimgen.src.modules.embedding.embedder import EmbedderSequential, EmbedderManager, BaseEmbedder
from stableclimgen.src.modules.rearrange import (RearrangeTimeCentric, RearrangeSpaceCentric,
                                                 RearrangeVarCentric)
from stableclimgen.src.modules.utils import EmbedBlock
from .. import embedding as embedding
from ...utils.helpers import check_value


class SelfAttention(nn.Module):
    """
    Self-attention layer with optional embeddings and causal mask support.

    This module implements the scaled dot-product attention mechanism with optional
    causal masking, suitable for time-series or sequence data.

    :param in_ch: Number of input channels.
    :param out_ch: Number of output channels.
    :param num_heads: Number of attention heads.
    :param dropout: Dropout probability for the output. Default is 0.
    :param is_causal: Whether to apply causal masking (True/False). Default is False.
    """

    def __init__(
            self,
            in_ch: int,
            out_ch: int,
            num_heads: int,
            dropout: float = 0.,
            is_causal: bool = False
    ):
        super().__init__()

        # Define query, key, and value projection layers
        self.to_q = torch.nn.Linear(in_ch, in_ch, bias=False)
        self.to_k = torch.nn.Linear(in_ch, in_ch, bias=True)
        self.to_v = torch.nn.Linear(in_ch, in_ch, bias=True)

        # Output projection layer to map the attention output to the desired channel size
        self.out_layer = torch.nn.Linear(in_ch, out_ch, bias=True)

        # Learnable scaling parameter to control the output's magnitude
        self.gamma = torch.nn.Parameter(torch.ones(out_ch) * 1E-6)

        self.n_heads = num_heads  # Number of attention heads
        self.dropout = dropout  # Dropout probability for attention output

        self.is_causal = is_causal  # Flag for causal masking (used in time-series tasks)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the SelfAttention layer.

        This function computes the attention mechanism using query, key, and value projections
        followed by applying the scaled dot-product attention with optional masking.

        :param x: Input tensor of shape (batch, time, channels).
        :param mask: Optional mask tensor (used for causal or attention masking).
        :return: Output tensor after applying self-attention and projection.
        """
        b, t_g_v, c = x.shape

        # Compute query, key, and value projections
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # Rearrange tensors for multi-head attention: [batch, time, (n_heads * d_head)] -> [batch, n_heads, time, d_head]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.n_heads), (q, k, v))

        # Apply scaled dot-product attention
        attn_out = self.scaled_dot_product_attention(q, k, v, mask)

        # Reshape and project output from multi-head attention back to the original dimensions
        attn_out = rearrange(attn_out, "b h t_g_v c -> b t_g_v (h c)", b=b, t_g_v=t_g_v, h=self.n_heads)

        # Apply skip connection and scaling to the output
        return self.gamma * self.out_layer(attn_out)

    def scaled_dot_product_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                     mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Scaled dot-product attention mechanism, with optional causal masking.

        :param q: Query tensor of shape [batch, n_heads, time, d_head].
        :param k: Key tensor of shape [batch, n_heads, time, d_head].
        :param v: Value tensor of shape [batch, n_heads, time, d_head].
        :param mask: Optional mask tensor for attention.
        :return: Output tensor after applying attention mechanism.
        """
        return scaled_dot_product_attention(q, k, v, mask, is_causal=self.is_causal)


class MLPLayer(nn.Module):
    """
    Multi-Layer Perceptron (MLP) layer with optional embedding and dropout.

    This MLP can be used in transformer blocks for nonlinear transformations with optional
    embedding layer and dropout regularization.

    :param in_ch: Number of input channels.
    :param out_ch: Number of output channels.
    :param mult: Multiplier for the hidden channels. Default is 1.
    :param dropout: Dropout probability for the hidden layers. Default is 0.
    """

    def __init__(
            self,
            in_ch: int,
            out_ch: int,
            mult: int = 1,
            dropout: float = 0.
    ):
        super().__init__()

        self.hidden_channels = in_ch * mult  # Hidden layer size

        # Define MLP with a hidden layer, GELU activation, and optional dropout
        self.branch_layer = torch.nn.Sequential(
            torch.nn.Linear(in_ch, self.hidden_channels),
            torch.nn.GELU(),
        )
        if dropout > 0.:
            self.branch_layer.append(torch.nn.Dropout(p=dropout))
        self.branch_layer.append(
            torch.nn.Linear(self.hidden_channels, out_ch)
        )

        # Learnable scaling parameter for the output of the MLP layer
        self.gamma = torch.nn.Parameter(torch.ones(out_ch) * 1E-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLPLayer.

        This function applies the MLP transformations and adds the skip connection to the output.

        :param x: Input tensor.
        :return: Output tensor after MLP transformation and skip connection.
        """
        return self.gamma * self.branch_layer(x)


class TransformerBlock(EmbedBlock):
    """
    Transformer block that combines MLP layers and Self-Attention layers with optional embeddings.

    This block applies a sequence of layers, such as attention or MLP, followed by normalization
    and optional embeddings (e.g., time, space, or variable embeddings) for each block.

    :param in_ch: Number of input channels.
    :param out_ch: List of output channels for each block in the sequence.
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
            in_ch: int,
            out_ch: List[int],
            blocks: List[str],
            seq_lengths: List[int] = None,
            embedder_names: List[List[str]] = None,
            embed_confs: Dict = None,
            embed_mode: str = "sum",
            num_heads: List[int] = 1,
            n_head_channels: List[int] = None,
            mlp_mult: List[int] = 1,
            dropout: List[float] = 0.,
            spatial_dim_count: int = 1,
            **kwargs
    ):
        super().__init__()

        if not out_ch:
            out_ch = in_ch  # Default output channels to input channels if not provided
        out_ch = check_value(out_ch, len(blocks))
        num_heads = check_value(num_heads, len(blocks))
        n_head_channels = check_value(n_head_channels, len(blocks))
        mlp_mult = check_value(mlp_mult, len(blocks))
        dropout = check_value(dropout, len(blocks))
        seq_lengths = check_value(seq_lengths, len(blocks))

        trans_blocks, embedders, embedding_layers, norms, residuals = [], [], [], [], []
        for i, block in enumerate(blocks):
            # Add MLP layer if specified
            if block == "mlp":
                trans_block = MLPLayer(in_ch, out_ch[i], mlp_mult[i], dropout[i])
                norm = torch.nn.LayerNorm(in_ch, elementwise_affine=False)
            else:
                # Select rearrangement function based on block type
                if block == "t":
                    rearrange_fn = RearrangeTimeCentric
                elif block == "s":
                    rearrange_fn = RearrangeSpaceCentric
                else:
                    assert block == "v"
                    rearrange_fn = RearrangeVarCentric
                n_heads = num_heads[i] if not n_head_channels[i] else in_ch // n_head_channels[i]
                trans_block = rearrange_fn(SelfAttention(in_ch, out_ch[i], n_heads,), spatial_dim_count, seq_lengths[i])
                norm = torch.nn.LayerNorm(in_ch, elementwise_affine=True)

            if embedder_names and embedder_names[i]:
                emb_dict = nn.ModuleDict()
                for emb_name in embedder_names[i]:
                    emb: BaseEmbedder = EmbedderManager().get_embedder(emb_name, **embed_confs[emb_name])
                    emb_dict[emb.name] = emb
                embedder_seq = EmbedderSequential(emb_dict, mode=embed_mode, spatial_dim_count = spatial_dim_count)
                embedding_layer = torch.nn.Linear(embedder_seq.get_out_channels, 2 * in_ch)

                embedders.append(embedder_seq)
                embedding_layers.append(embedding_layer)
            else:
                embedders.append(None)
                embedding_layers.append(None)

            # Skip connection layer: Identity if in_ch == out_ch, else a linear projection
            if in_ch != out_ch[i]:
                residual = torch.nn.Linear(in_ch, out_ch[i])
            else:
                residual = torch.nn.Identity()

            # append normalization and trans_block
            trans_blocks.append(trans_block)
            norms.append(norm)
            residuals.append(residual)

            in_ch = out_ch if isinstance(out_ch, int) else out_ch[i]

        self.embedders = nn.ModuleList(embedders)
        self.embedding_layers = nn.ModuleList(embedding_layers)
        self.blocks = nn.ModuleList(trans_blocks)
        self.norms = nn.ModuleList(norms)
        self.residuals = nn.ModuleList(residuals)

    def forward(self, x: torch.Tensor, emb: Optional[Dict] = None, mask: Optional[torch.Tensor] = None,
                cond: Optional[torch.Tensor] = None, *args) -> torch.Tensor:
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
        for norm, block, embedder, embedding_layer, residual in zip(self.norms, self.blocks, self.embedders, self.embedding_layers, self.residuals):
            if embedder:
                # Apply the embedding transformation (scale and shift)
                scale, shift = embedding_layer(embedder(emb)).chunk(2, dim=-1)
                out = norm(x) * (scale + 1) + shift
            else:
                out = norm(x)
            x = block(out) + residual(x)
        return x
