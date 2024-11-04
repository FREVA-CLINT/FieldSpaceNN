from abc import abstractmethod

import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

from stableclimgen.src.modules.cnn.resnet import zero_module
from stableclimgen.src.modules.transformer.transformer_base import EmbedBlock


class PatchEmbedder3D(EmbedBlock):
    def __init__(self, in_channels, out_channels, kernel_size, patch_size):
        super().__init__()
        assert len(patch_size) == len(kernel_size) == 3
        padding = tuple((kernel_size[i] - patch_size[i])//2 for i in range(len(kernel_size)))
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, patch_size, padding=padding)

    def forward(self, x, emb=None, mask=None, cond=None, coords=None, **kwargs):
        x = self.conv(x)
        return x


class ConvUnpatchify(EmbedBlock):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            zero_module(nn.Conv3d(in_channels, out_channels, 3, padding=1))
        )

    def forward(self, x, emb=None, mask=None, cond=None, coords=None, out_shape=None):
        x = F.interpolate(x, size=out_shape, mode="nearest")
        return self.out(x)


class LinearUnpatchify(EmbedBlock):
    """
    The final layer of DiT.
    """

    def __init__(self, in_channels, out_channels, patch_size, embed_dim):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(in_channels, patch_size[0] * patch_size[1] * patch_size[2] * out_channels, bias=True)
        self.patch_size = patch_size
        self.out_channels = out_channels
        if embed_dim is not None:
            self.embedding_layer = nn.Sequential(
                nn.SiLU(),
                nn.Linear(embed_dim, 2 * in_channels, bias=True)
            )

    def forward(self, x, emb=None, mask=None, cond=None, coords=None, out_shape=None):
        b, v= x.shape[0], x.shape[1]
        x = rearrange(x, 'b v ... c -> (b v) (...) c')
        if hasattr(self, 'embedding_layer'):
            emb = rearrange(emb, 'b v ... c -> (b v) (...) c')
            scale, shift = self.embedding_layer(emb).chunk(2, dim=-1)
            x = self.norm(x) * (scale + 1) + shift
        else:
            x = self.norm(x)
        x = self.linear(x)
        return rearrange(self.unpatchify(x, out_shape), '(b v) ... -> b v ...', b=b, v=v)

    def unpatchify(self, x, out_shape):
        c = self.out_channels
        p = self.patch_size
        t, h, w = out_shape
        x = x.reshape(shape=(x.shape[0], t, h, w, p[0], p[1], p[2], c))
        x = torch.einsum('nthwpqrc->nctphqwr', x)
        return x.reshape(shape=(x.shape[0], c, t * p[0], h * p[1], w * p[2]))