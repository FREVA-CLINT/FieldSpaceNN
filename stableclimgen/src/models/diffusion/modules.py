from abc import abstractmethod

import torch
import torch.nn as nn
from einops import rearrange

from .nn import (
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization
)


class EmbedBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb, mask, cond):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class EmbedBlockSequential(nn.Sequential, EmbedBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb=None, mask=None, cond=None):
        for layer in self:
            if isinstance(layer, EmbedBlock):
                x = layer(x, emb, mask, cond)
            else:
                x = layer(x)
        return x


class SpatioTemporalPatchEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, patch_size):
        super().__init__()
        assert len(patch_size) == len(kernel_size) == 3
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, patch_size)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param in_channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, in_channels, upsample, use_conv=True, dims=2, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.use_conv = use_conv
        self.dims = dims
        self.upsample = upsample
        if use_conv:
            self.conv = conv_nd(dims, self.in_channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.in_channels
        x = self.upsample(x)
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param in_channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, in_channels, use_conv=True, dims=2, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.in_channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.in_channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.in_channels
        return self.op(x)


class ConvBlock(EmbedBlock):
    def __init__(self, in_channels, out_channels, up=False, down=False, img_size=None):
        super().__init__()
        if up:
            self.conv = Upsample(in_channels, nn.Upsample(img_size, mode="nearest"), out_channels=out_channels)
        elif down:
            self.conv = Downsample(in_channels, out_channels=out_channels)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)

    def forward(self, x, emb=None, mask=None, cond=None):
        return self.conv(x)


class ResBlock(EmbedBlock):
    """
    A residual block that can optionally change the number of channels.

    :param in_channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
            self,
            in_channels,
            emb_channels,
            dropout,
            out_channels=None,
            use_conv=False,
            use_scale_shift_norm=False,
            dims=2,
            up=False,
            down=False,
            img_size=None
    ):
        super().__init__()
        self.in_channels = in_channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or in_channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(in_channels),
            nn.SiLU(),
            conv_nd(dims, in_channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(in_channels, nn.Upsample(img_size, mode="nearest"), True, dims)
            self.x_upd = Upsample(in_channels, nn.Upsample(img_size, mode="nearest"), True, dims)
        elif down:
            self.h_upd = Downsample(in_channels, True, dims)
            self.x_upd = Downsample(in_channels, True, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == in_channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, in_channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, in_channels, self.out_channels, 1)

    def forward(self, x, emb=None, mask=None, cond=None):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb = emb[:, :, 0, 0]
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class Identity(EmbedBlock):
    def __init__(self):
        super().__init__()
        self.fn = torch.nn.Identity()

    def forward(self, x, emb=None, mask=None, cond=None, **kwargs):
        return self.fn(x)


class FinalLayer(EmbedBlock):
    """
    The final layer of DiT.
    """

    def __init__(self, in_channels, out_channels, patch_size, embed_dim, img_size, seq_length):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(in_channels, patch_size[0] * patch_size[1] * patch_size[2] * out_channels, bias=True)
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.img_size = img_size
        self.seq_length = seq_length
        if embed_dim is not None:
            self.embedding_layer = nn.Sequential(
                nn.SiLU(),
                nn.Linear(embed_dim, 2 * in_channels, bias=True)
            )

    def forward(self, x, emb, mask, cond):
        b, c, t, w, h = x.shape
        x = rearrange(x, 'b c t w h -> b (t w h) c')
        if hasattr(self, 'embedding_layer'):
            emb = rearrange(emb, 'b c t w h -> b (t w h) c')
            scale, shift = self.embedding_layer(emb).chunk(2, dim=-1)
            x = self.norm(x) * (scale + 1) + shift
        else:
            x = self.norm(x)
        x = self.linear(x)
        return self.unpatchify(x, b)

    def unpatchify(self, x, b):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.patch_size
        h, w = self.img_size
        t = x.shape[1] // (h * w)
        x = x.reshape(shape=(x.shape[0], t, h, w, p[0], p[1], p[2], c))
        x = torch.einsum('nthwpqrc->nctphqwr', x)
        return x.reshape(shape=(x.shape[0], c, t * p[0], h * p[1], w * p[2]))