from abc import abstractmethod
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

from stableclimgen.src.modules.cnn.resnet import zero_module
from stableclimgen.src.modules.transformer.transformer_base import EmbedBlock
from typing import Tuple, Optional


class PatchEmbedder3D(EmbedBlock):
    """
    Embeds a 3D patch from the input using a convolutional layer.

    :param in_channels: Number of input channels.
    :param out_channels: Number of output channels.
    :param kernel_size: Size of the convolutional kernel as a tuple of three integers.
    :param patch_size: Size of the patch as a tuple of three integers.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int, int], patch_size: Tuple[int, int, int]):
        super().__init__()
        assert len(patch_size) == len(kernel_size) == 3
        padding = tuple((kernel_size[i] - patch_size[i]) // 2 for i in range(len(kernel_size)))
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, patch_size, padding=padding)

    def forward(self, x: torch.Tensor, emb: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None,
                cond: Optional[torch.Tensor] = None, coords: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        Forward pass through the patch embedding layer.

        :param x: Input tensor.
        :param emb: Optional embedding tensor.
        :param mask: Optional mask tensor.
        :param cond: Optional conditioning tensor.
        :param coords: Optional coordinates tensor.
        :return: Embedded output tensor.
        """
        x = self.conv(x)
        return x


class ConvUnpatchify(EmbedBlock):
    """
    Converts patches back to the original shape using a convolutional layer.

    :param in_channels: Number of input channels.
    :param out_channels: Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.out = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            zero_module(nn.Conv3d(in_channels, out_channels, 3, padding=1))
        )

    def forward(self, x: torch.Tensor, emb: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None,
                cond: Optional[torch.Tensor] = None, coords: Optional[torch.Tensor] = None, out_shape: Optional[Tuple[int, int, int]] = None) -> torch.Tensor:
        """
        Forward pass through the unpatchifying layer with interpolation.

        :param x: Input tensor.
        :param emb: Optional embedding tensor.
        :param mask: Optional mask tensor.
        :param cond: Optional conditioning tensor.
        :param coords: Optional coordinates tensor.
        :param out_shape: Target output shape for interpolation.
        :return: Reconstructed output tensor.
        """
        x = F.interpolate(x, size=out_shape, mode="nearest")
        return self.out(x)


class LinearUnpatchify(EmbedBlock):
    """
    Converts linear embeddings back to the original shape.

    :param in_channels: Number of input channels.
    :param out_channels: Number of output channels.
    :param patch_size: Size of the patches as a tuple of three integers.
    :param embed_dim: Embedding dimension, if applicable.
    """

    def __init__(self, in_channels: int, out_channels: int, patch_size: Tuple[int, int, int], embed_dim: Optional[int] = None):
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

    def forward(self, x: torch.Tensor, emb: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None,
                cond: Optional[torch.Tensor] = None, coords: Optional[torch.Tensor] = None, out_shape: Optional[Tuple[int, int, int]] = None) -> torch.Tensor:
        """
        Forward pass to reconstruct the linear embeddings back to original dimensions.

        :param x: Input tensor.
        :param emb: Optional embedding tensor for scale-shift normalization.
        :param mask: Optional mask tensor.
        :param cond: Optional conditioning tensor.
        :param coords: Optional coordinates tensor.
        :param out_shape: Target output shape for the unpatchifying operation.
        :return: Reconstructed output tensor.
        """
        b, v = x.shape[0], x.shape[1]
        x = rearrange(x, 'b v ... c -> (b v) (...) c')
        if hasattr(self, 'embedding_layer'):
            emb = rearrange(emb, 'b v ... c -> (b v) (...) c')
            scale, shift = self.embedding_layer(emb).chunk(2, dim=-1)
            x = self.norm(x) * (scale + 1) + shift
        else:
            x = self.norm(x)
        x = self.linear(x)
        return rearrange(self.unpatchify(x, out_shape), '(b v) ... -> b v ...', b=b, v=v)

    def unpatchify(self, x: torch.Tensor, out_shape: Tuple[int, int, int]) -> torch.Tensor:
        """
        Reshape the output tensor to the original dimensions after linear unpatchifying.

        :param x: Input tensor to be reshaped.
        :param out_shape: Target output shape as a tuple of three integers.
        :return: Tensor reshaped to the original dimensions.
        """
        c = self.out_channels
        p = self.patch_size
        t, h, w = out_shape
        x = x.reshape(shape=(x.shape[0], t, h, w, p[0], p[1], p[2], c))
        x = torch.einsum('nthwpqrc->nctphqwr', x)
        return x.reshape(shape=(x.shape[0], c, t * p[0], h * p[1], w * p[2]))
