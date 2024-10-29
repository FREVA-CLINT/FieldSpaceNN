import math

import torch
import torch.nn as nn
from einops import rearrange

from .modules import EmbedBlock


class RelativePosition1D(nn.Module):
    """ https://github.com/evelinehong/Transformer_Relative_Position_PyTorch/blob/master/relative_position.py """

    def __init__(self, n_heads, window_size, max_window_size=None):
        super().__init__()
        self.n_heads = n_heads
        self.max_window_size = window_size if max_window_size is None else max_window_size

        self.embeddings_table = nn.Parameter(torch.Tensor(max_window_size * 2 + 1, n_heads))
        nn.init.xavier_uniform_(self.embeddings_table)
        self.activation = nn.SiLU()

    def forward(self, context_win, memory_win):
        context_win = torch.arange(context_win)
        memory_win = torch.arange(memory_win)
        distance_mat = memory_win[None, :] - context_win[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_window_size, self.max_window_size)
        relative_position = distance_mat_clipped + self.max_window_size
        relative_position_index = relative_position.long()
        embeddings = self.embeddings_table[relative_position_index]
        return rearrange(self.activation(embeddings), "t s d -> d t s")


class RelativePosition2D(nn.Module):
    def __init__(self, num_heads, max_window_size, window_size=None):
        super(RelativePosition2D, self).__init__()
        self.num_heads = num_heads
        self.max_width = max_window_size[0]
        self.max_height = max_window_size[1]

        self.width = self.max_width if window_size is None else window_size[0]
        self.height = self.max_height if window_size is None else window_size[1]

        coords_w = torch.arange(self.width)
        coords_h = torch.arange(self.height)

        coords = torch.stack(torch.meshgrid([coords_w, coords_h]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.max_width - 1
        relative_coords[:, :, 1] += self.max_height - 1
        relative_coords[:, :, 0] *= 2 * self.max_height - 1

        self.relative_position_index = relative_coords.sum(-1).clamp(
            0, (2 * self.max_height - 1) * (2 * self.max_width - 1) - 1)

        self.embeddings_table = nn.Parameter(
            torch.zeros((2 * self.max_height - 1) * (2 * self.max_width - 1), num_heads)
        )
        nn.init.xavier_uniform_(self.embeddings_table)

        self.activation = nn.SiLU()

    def forward(self, context_win, memory_win):
        embeddings = self.embeddings_table[self.relative_position_index.view(-1)].view(self.width * self.height, self.width * self.height, -1)
        return rearrange(self.activation(embeddings), "t s d -> d t s")


class SelfAttention(EmbedBlock):
    """
    Self-Attention layer with implementation inspired by
    https://github.com/Stability-AI/generative-models/
    """

    def __init__(
            self,
            channels,
            cond_channels,
            num_heads,
            num_head_channels=-1,
            dropout=0.,
            embed_dim=None,
            rel_position=None,
            window_size=None,
            max_window_size=None
    ):
        super().__init__()
        if num_head_channels != -1 and channels % num_head_channels == 0:
            num_heads = channels // num_head_channels
        if cond_channels is None:
            cond_channels = channels

        if embed_dim:
            self.embedding_layer = torch.nn.Linear(embed_dim, 2 * channels)
        if rel_position:
            rel_pos = RelativePosition2D if isinstance(window_size, tuple) else RelativePosition1D
            if rel_position == "bias":
                self.rel_pos_b = rel_pos(num_heads, window_size, max_window_size)
            elif rel_position == "context":
                self.rel_pos_q = rel_pos(num_heads, window_size, max_window_size)
                self.rel_pos_k = rel_pos(num_heads, window_size, max_window_size)
                self.rel_pos_v = rel_pos(num_heads, window_size, max_window_size)

        self.rel_position = rel_position
        self.norm = torch.nn.LayerNorm(channels, elementwise_affine=True)
        self.norm_cond = torch.nn.LayerNorm(channels, elementwise_affine=True)
        self.to_q = torch.nn.Linear(channels, channels, bias=False)
        self.to_k = torch.nn.Linear(cond_channels, channels, bias=True)
        self.to_v = torch.nn.Linear(cond_channels, channels, bias=True)

        self.out_layer = torch.nn.Linear(channels, channels, bias=True)
        self.gamma = torch.nn.Parameter(torch.ones(channels) * 1E-6)
        self.n_heads = num_heads
        self.dropout = dropout

    def forward(self, x, emb=None, mask=None, cond=None, is_causal=False):
        b, t_g, c = x.shape
        # In-/Output dimensions are: (batch size, grid points/time, channels)
        if hasattr(self, 'embedding_layer'):
            scale, shift = self.embedding_layer(emb).chunk(2, dim=-1)
            out = self.norm(x) * (scale + 1) + shift
        else:
            out = self.norm(x)

        if not torch.is_tensor(cond):
            cond = out
        else:
            cond = self.norm_cond(cond)

        q = self.to_q(out)
        k = self.to_k(cond)
        v = self.to_v(cond)

        # Split into q k v (k), heads (h) and channels (c).
        # Move time into batch dimension.
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.n_heads), (q, k, v))

        # Use of scaled dot product attention function from pytorch
        attn_out = self.scaled_dot_product_attention(q, k, v, mask, is_causal)

        # Rearrange to get correct output again
        attn_out = rearrange(
            attn_out, "b h t_g c -> b t_g (h c)",
            b=b, t_g=t_g, h=self.n_heads
        )
        # Gamma scaling is initialized to a small number, so that at init it is
        # roughly an identity
        return x + self.gamma * self.out_layer(attn_out)

    def scaled_dot_product_attention(self, q, k, v, mask=None, is_causal=False):
        l, s = q.size(-2), k.size(-2)
        scale_factor = 1 / math.sqrt(q.size(-1))
        attn_weight = q @ k.transpose(-2, -1) * scale_factor

        if is_causal:
            assert mask is None
            attn_bias = torch.zeros(l, s, dtype=q.dtype, device=q.device)
            temp_mask = torch.ones(l, s, dtype=torch.bool, device=q.device).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(q.dtype)
            attn_weight += attn_bias

        # mask attention
        if mask is not None:
            max_neg_value = -1e9
            temp_mask = (1 - mask[:, :, 0].float())[:, None, :, None]
            attn_weight += temp_mask * max_neg_value

        # Add relative position encoding
        if self.rel_position == "bias":
            r = self.rel_pos_b(q.shape[2], k.shape[2])
            attn_weight += r.unsqueeze(0)
        elif self.rel_position == "context":
            r_q = self.rel_pos_q(q.shape[2], k.shape[2])
            r_k = self.rel_pos_k(q.shape[2], k.shape[2])
            r_q = torch.einsum('bhgc,hgt->bhgt', k, r_q)
            r_k = torch.einsum('bhgc,hgt->bhgt', q, r_k)
            attn_weight += r_q + r_k

        attn_weight = torch.softmax(attn_weight, dim=-1)
        if not self.training and self.dropout > 0.0:
            attn_weight = torch.dropout(attn_weight, p=self.dropout, train=self.training)

        attn_weight = attn_weight @ v

        if self.rel_position == "context":
            r_v = self.rel_pos_v(q.shape[2], k.shape[2])
            r_v = torch.einsum('bhgc,hgt->bhgc', attn_weight, r_v)
            attn_weight = attn_weight + r_v

        return attn_weight


class MLPLayer(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            mult=1,
            dropout_rate=0.,
            embed_dim=None
    ):
        super().__init__()
        if embed_dim:
            self.embedding_layer = torch.nn.Linear(embed_dim, 2 * in_channels)
        self.hidden_channels = in_channels * mult
        self.norm = torch.nn.LayerNorm(in_channels, elementwise_affine=False)
        self.branch_layer = torch.nn.Sequential(
            torch.nn.Linear(in_channels, self.hidden_channels),
            torch.nn.GELU(),
        )
        if dropout_rate > 0.:
            self.branch_layer.append(torch.nn.Dropout(p=dropout_rate))
        self.branch_layer.append(
            torch.nn.Linear(self.hidden_channels, out_channels)
        )

        if in_channels != out_channels:
            self.skip_connection = torch.nn.Linear(in_channels, out_channels)
        else:
            self.skip_connection = torch.nn.Identity()

        self.gamma = torch.nn.Parameter(torch.ones(out_channels) * 1E-6)

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
            in_channels,
            out_channels,
            spatial_attention,
            temporal_attention,
            spatial_cross_attention=None,
            temporal_cross_attention=None,
            time_window=None,
            mult=1,
            dropout=0.,
            embed_dim=None,
            time_causal=False
    ):
        super().__init__()
        self.spatial_attention = spatial_attention
        self.temporal_attention = temporal_attention
        self.spatial_cross_attention = spatial_cross_attention
        self.temporal_cross_attention = temporal_cross_attention
        self.out_channels = out_channels
        self.mlp = MLPLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            mult=mult,
            dropout_rate=dropout,
            embed_dim=embed_dim
        )
        self.time_window = time_window
        self.time_causal = time_causal

    def forward(self, x, emb=None, mask=None, cond=None):
        # masked spatio-temporal attention
        x = self.spatial_attention(x, emb=emb, mask=mask)
        x = self.temporal_attention(x, emb=emb, mask=mask if not self.time_causal else None,
                                    is_causal=self.time_causal)

        # unmasked cross attention
        if torch.is_tensor(cond):
            x = self.spatial_cross_attention(x, cond=cond)
            x = self.temporal_cross_attention(x, cond=cond)

        b, _, t, w, h = x.shape

        out_tensor = rearrange(x, 'b c t w h -> b t (w h) c')
        emb = rearrange(emb, 'b c t w h -> b t (w h) c')

        out_tensor = self.mlp(out_tensor, emb)

        # Use einops.rearrange to reshape the tensor
        out_tensor = rearrange(out_tensor, 'b t (w h) c -> b c t w h', w=w, h=h)
        return out_tensor