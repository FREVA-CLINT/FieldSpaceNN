from abc import abstractmethod
from dbm import error
from typing import Union, List

import torch
from einops import rearrange
from torch import nn
from torch.nn.functional import scaled_dot_product_attention

from stableclimgen.src.modules.transformer.base_rearrange import RearrangeBlock, RearrangeTimeCentric, \
    RearrangeSpaceCentric, RearrangeVarCentric


class EmbedBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb, mask, cond, coords):
        """
        Apply the module to `x` given `emb`, `mask`, `cond`, `coords`.
        """

class EmbedBlockSequential(nn.Sequential, EmbedBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb=None, mask=None, cond=None, coords=None):
        for layer in self:
            if isinstance(layer, EmbedBlock):
                x = layer(x, emb, mask, cond, coords)
            else:
                x = layer(x)
        return x


class SelfAttention(EmbedBlock):
    """
    Self-Attention layer with implementation inspired by
    https://github.com/Stability-AI/generative-models/
    """

    def __init__(
            self,
            in_ch,
            out_ch,
            num_heads,
            dropout=0.,
            embed_dim=None,
            is_causal=False
    ):
        super().__init__()
        if embed_dim:
            self.embedding_layer = torch.nn.Linear(embed_dim, 2 * in_ch)

        self.norm = torch.nn.LayerNorm(in_ch, elementwise_affine=True)
        self.to_q = torch.nn.Linear(in_ch, in_ch, bias=False)
        self.to_k = torch.nn.Linear(in_ch, in_ch, bias=True)
        self.to_v = torch.nn.Linear(in_ch, in_ch, bias=True)

        self.out_layer = torch.nn.Linear(in_ch, out_ch, bias=True)
        self.gamma = torch.nn.Parameter(torch.ones(out_ch) * 1E-6)
        self.n_heads = num_heads
        self.dropout = dropout

        if in_ch != out_ch:
            self.skip_connection = torch.nn.Linear(in_ch, out_ch)
        else:
            self.skip_connection = torch.nn.Identity()

        self.is_causal = is_causal

    def forward(self, x, emb=None, mask=None, cond=None, coords=None):
        b, t_g, c = x.shape
        # In-/Output dimensions are: (batch size, grid points/time, channels)
        if hasattr(self, 'embedding_layer'):
            scale, shift = self.embedding_layer(emb).chunk(2, dim=-1)
            out = self.norm(x) * (scale + 1) + shift
        else:
            out = self.norm(x)

        q = self.to_q(out)
        k = self.to_k(out)
        v = self.to_v(out)

        # Split into q k v (k), heads (h) and channels (c).
        # Move time into batch dimension.
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.n_heads), (q, k, v))

        # Use of scaled dot product attention function from pytorch
        attn_out = self.scaled_dot_product_attention(q, k, v, mask)

        # Rearrange to get correct output again
        attn_out = rearrange(
            attn_out, "b h t_g_v c -> b t_g_v (h c)",
            b=b, t_g=t_g, h=self.n_heads
        )
        # Gamma scaling is initialized to a small number, so that at init it is
        # roughly an identity
        return self.skip_connection(x) + self.gamma * self.out_layer(attn_out)

    def scaled_dot_product_attention(self, q, k, v, mask):
        return scaled_dot_product_attention(q, k, v, mask, is_causal=self.is_causal)


class MLPLayer(torch.nn.Module):
    def __init__(
            self,
            in_ch,
            out_ch,
            mult=1,
            dropout=0.,
            embed_dim=None
    ):
        super().__init__()
        if embed_dim:
            self.embedding_layer = torch.nn.Linear(embed_dim, 2 * in_ch)
        self.hidden_channels = in_ch * mult
        self.norm = torch.nn.LayerNorm(in_ch, elementwise_affine=False)
        self.branch_layer = torch.nn.Sequential(
            torch.nn.Linear(in_ch, self.hidden_channels),
            torch.nn.GELU(),
        )
        if dropout > 0.:
            self.branch_layer.append(torch.nn.Dropout(p=dropout))
        self.branch_layer.append(
            torch.nn.Linear(self.hidden_channels, out_ch)
        )

        if in_ch != out_ch:
            self.skip_connection = torch.nn.Linear(in_ch, out_ch)
        else:
            self.skip_connection = torch.nn.Identity()

        self.gamma = torch.nn.Parameter(torch.ones(out_ch) * 1E-6)

    def forward(self, x, emb=None):
        if hasattr(self, 'embedding_layer'):
            scale, shift = self.embedding_layer(emb).chunk(2, dim=-1)
            branch = self.norm(x) * (scale + 1) + shift
        else:
            branch = self.norm(x)
        return self.skip_connection(x) + self.gamma * self.branch_layer(branch)


class TransformerBlock(EmbedBlock):
    def __init__(
            self,
            blocks: List[str],
            in_ch: Union[int, List[int]],
            out_ch: Union[int, List[int]] = None,
            num_heads: Union[int, List[int]] = 1,
            mlp_mult: Union[int, List[int]] = 1,
            embed_dim: Union[int, List[int]] = None,
            dropout: Union[float, List[float]]=0.
    ):
        super().__init__()
        if not out_ch:
            out_ch = in_ch
        print(blocks)
        trans_blocks = []
        for block in blocks:
            if block == "mlp":
                trans_blocks.append(MLPLayer(
                    in_ch if isinstance(in_ch, int) else in_ch.pop(0),
                    out_ch if isinstance(out_ch, int) else out_ch.pop(0),
                    mlp_mult if isinstance(mlp_mult, int) else mlp_mult.pop(0),
                    embed_dim if isinstance(embed_dim, int) else embed_dim.pop(0),
                    dropout if isinstance(dropout, float) else dropout.pop(0)
                ))
            else:
                if block == "t":
                    rearrange_fn = RearrangeTimeCentric
                elif block == "s":
                    rearrange_fn = RearrangeSpaceCentric
                else:
                    assert block == "v"
                    rearrange_fn = RearrangeVarCentric
                trans_blocks.append(rearrange_fn(SelfAttention(
                    in_ch if isinstance(in_ch, int) else in_ch.pop(0),
                    out_ch if isinstance(out_ch, int) else out_ch.pop(0),
                    num_heads if isinstance(num_heads, int) else num_heads.pop(0),
                    embed_dim if isinstance(embed_dim, int) else embed_dim.pop(0),
                )))

        self.blocks = EmbedBlockSequential(*trans_blocks)

    def forward(self, x, emb=None, mask=None, cond=None, coords=None):
        # masked spatio-temporal attention
        for block in self.blocks:
            x = block(x, emb, mask, cond, coords)
        return x