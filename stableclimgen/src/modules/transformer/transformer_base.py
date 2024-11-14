from typing import Union, List, Optional

import torch
from einops import rearrange
from torch.nn.functional import scaled_dot_product_attention

from stableclimgen.src.modules.rearrange import (RearrangeTimeCentric, RearrangeSpaceCentric,
                                                 RearrangeVarCentric)
from stableclimgen.src.modules.utils import EmbedBlock, EmbedBlockSequential


class SelfAttention(EmbedBlock):
    """
    Self-attention layer with optional embeddings and causal mask support.

    :param in_ch: Number of input channels.
    :param out_ch: Number of output channels.
    :param num_heads: Number of attention heads.
    :param dropout: Dropout probability. Default is 0.
    :param embed_dim: Dimension of embeddings. If provided, an embedding layer is applied.
    :param is_causal: Whether to use causal masking. Default is False.
    """

    def __init__(
            self,
            in_ch: int,
            out_ch: int,
            num_heads: int,
            dropout: float = 0.,
            embed_dim: Optional[int] = None,
            is_causal: bool = False
    ):
        super().__init__()
        # Optional embedding layer for scale-shift normalization
        if embed_dim:
            self.embedding_layer = torch.nn.Linear(embed_dim, 2 * in_ch)

        self.norm = torch.nn.LayerNorm(in_ch, elementwise_affine=True)  # Layer normalization
        # Define query, key, and value projection layers
        self.to_q = torch.nn.Linear(in_ch, in_ch, bias=False)
        self.to_k = torch.nn.Linear(in_ch, in_ch, bias=True)
        self.to_v = torch.nn.Linear(in_ch, in_ch, bias=True)

        self.out_layer = torch.nn.Linear(in_ch, out_ch, bias=True)  # Output projection layer
        self.gamma = torch.nn.Parameter(torch.ones(out_ch) * 1E-6)  # Scaling parameter
        self.n_heads = num_heads  # Number of attention heads
        self.dropout = dropout

        # Define skip connection layer
        if in_ch != out_ch:
            self.skip_connection = torch.nn.Linear(in_ch, out_ch)
        else:
            self.skip_connection = torch.nn.Identity()

        self.is_causal = is_causal  # Causal masking flag

    def forward(self, x: torch.Tensor, emb: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None,
                cond: Optional[torch.Tensor] = None, coords: Optional[torch.Tensor] = None, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass for the SelfAttention layer.

        :param x: Input tensor of shape (batch, time, channels).
        :param emb: Optional embedding tensor.
        :param mask: Optional mask tensor.
        :param cond: Optional conditioning tensor.
        :param coords: Optional coordinates tensor.
        :return: Output tensor with self-attention applied.
        """
        b, t_g_v, c = x.shape
        # Apply scale-shift normalization if embedding is provided
        if hasattr(self, 'embedding_layer'):
            scale, shift = self.embedding_layer(emb).chunk(2, dim=-1)
            out = self.norm(x) * (scale + 1) + shift
        else:
            out = self.norm(x)

        # Compute query, key, and value projections
        q = self.to_q(out)
        k = self.to_k(out)
        v = self.to_v(out)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.n_heads), (q, k, v))

        # Apply scaled dot-product attention
        attn_out = self.scaled_dot_product_attention(q, k, v, mask)

        # Reshape and project output
        attn_out = rearrange(attn_out, "b h t_g_v c -> b t_g_v (h c)", b=b, t_g_v=t_g_v, h=self.n_heads)
        return self.skip_connection(x) + self.gamma * self.out_layer(attn_out)

    def scaled_dot_product_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                     mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Scaled dot-product attention mechanism.

        :param q: Query tensor.
        :param k: Key tensor.
        :param v: Value tensor.
        :param mask: Optional mask tensor.
        :return: Output tensor after applying attention.
        """
        return scaled_dot_product_attention(q, k, v, mask, is_causal=self.is_causal)


class MLPLayer(EmbedBlock):
    """
    Multi-Layer Perceptron (MLP) layer with optional embedding.

    :param in_ch: Number of input channels.
    :param out_ch: Number of output channels.
    :param mult: Multiplier for the hidden channels. Default is 1.
    :param embed_dim: Dimension of embeddings. If provided, an embedding layer is applied.
    :param dropout: Dropout probability. Default is 0.
    """

    def __init__(
            self,
            in_ch: int,
            out_ch: int,
            mult: int = 1,
            embed_dim: Optional[int] = None,
            dropout: float = 0.
    ):
        super().__init__()
        if embed_dim:
            self.embedding_layer = torch.nn.Linear(embed_dim, 2 * in_ch)
        self.hidden_channels = in_ch * mult
        self.norm = torch.nn.LayerNorm(in_ch, elementwise_affine=False)
        # Define MLP with optional dropout
        self.branch_layer = torch.nn.Sequential(
            torch.nn.Linear(in_ch, self.hidden_channels),
            torch.nn.GELU(),
        )
        if dropout > 0.:
            self.branch_layer.append(torch.nn.Dropout(p=dropout))
        self.branch_layer.append(
            torch.nn.Linear(self.hidden_channels, out_ch)
        )

        # Define skip connection layer
        if in_ch != out_ch:
            self.skip_connection = torch.nn.Linear(in_ch, out_ch)
        else:
            self.skip_connection = torch.nn.Identity()

        self.gamma = torch.nn.Parameter(torch.ones(out_ch) * 1E-6)  # Scaling parameter

    def forward(self, x: torch.Tensor, emb: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None,
                cond: Optional[torch.Tensor] = None, coords: Optional[torch.Tensor] = None, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass for the MLPLayer.

        :param x: Input tensor.
        :param emb: Optional embedding tensor.
        :param mask: Optional mask tensor.
        :param cond: Optional conditioning tensor.
        :param coords: Optional coordinates tensor.
        :return: Output tensor after applying the MLP.
        """
        # Apply scale-shift normalization if embedding is provided
        if hasattr(self, 'embedding_layer'):
            scale, shift = self.embedding_layer(emb).chunk(2, dim=-1)
            branch = self.norm(x) * (scale + 1) + shift
        else:
            branch = self.norm(x)
        return self.skip_connection(x) + self.gamma * self.branch_layer(branch)


class TransformerBlock(EmbedBlock):
    """
    Transformer block with multiple sub-layers including MLP and Self-Attention layers.

    :param in_ch: Number of input channels.
    :param out_ch: Number of output channels.
    :param blocks: List of blocks to include, either "mlp" or "t"/"s"/"v" for self-attention with time/space/variable rearrangement.
    :param num_heads: Number of attention heads for self-attention blocks.
    :param mlp_mult: Multiplier for the hidden channels in MLP blocks.
    :param embed_dim: Dimension of embeddings.
    :param dropout: Dropout probability for each block.
    :param spatial_dim_count: Determines the number of spatial dimensions
    """

    def __init__(
            self,
            in_ch: int,
            out_ch: Union[int, List[int]],
            blocks: Union[str, List[str]],
            num_heads: Union[int, List[int]] = 1,
            mlp_mult: Union[int, List[int]] = 1,
            embed_dim: Optional[Union[int, List[int]]] = None,
            dropout: Union[float, List[float]] = 0.,
            spatial_dim_count: int = 1,
            **kwargs
    ):
        super().__init__()
        if not out_ch:
            out_ch = in_ch  # Default output channels to input channels if not provided
        trans_blocks = []
        if isinstance(blocks, str):
            blocks = [blocks]  # Ensure blocks is a list
        for i, block in enumerate(blocks):
            # Add MLP layer if specified
            if block == "mlp":
                trans_blocks.append(MLPLayer(
                    in_ch,
                    out_ch if isinstance(out_ch, int) else out_ch[i],
                    mlp_mult if isinstance(mlp_mult, int) else mlp_mult.pop(0),
                    embed_dim if isinstance(embed_dim, int) or embed_dim is None else embed_dim.pop(0),
                    dropout if isinstance(dropout, float) else dropout.pop(0)
                ))
            else:
                # Select rearrangement function based on block type
                if block == "t":
                    rearrange_fn = RearrangeTimeCentric
                elif block == "s":
                    rearrange_fn = RearrangeSpaceCentric
                else:
                    assert block == "v"
                    rearrange_fn = RearrangeVarCentric
                trans_blocks.append(rearrange_fn(SelfAttention(
                    in_ch,
                    out_ch if isinstance(out_ch, int) else out_ch[i],
                    num_heads if isinstance(num_heads, int) else num_heads.pop(0),
                    embed_dim if isinstance(embed_dim, int) or embed_dim is None else embed_dim.pop(0),
                ), spatial_dim_count))
            in_ch = out_ch if isinstance(out_ch, int) else out_ch[i]

        self.blocks = EmbedBlockSequential(*trans_blocks)

    def forward(self, x: torch.Tensor, emb: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None,
                cond: Optional[torch.Tensor] = None, coords: Optional[torch.Tensor] = None, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass for the TransformerBlock.

        :param x: Input tensor.
        :param emb: Optional embedding tensor.
        :param mask: Optional mask tensor.
        :param cond: Optional conditioning tensor.
        :param coords: Optional coordinates tensor.
        :return: Output tensor after applying transformer blocks sequentially.
        """
        for block in self.blocks:
            x = block(x, emb, mask, cond, coords)
        return x
