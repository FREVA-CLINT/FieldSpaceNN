from typing import Any, List, Optional

import torch
import torch.nn as nn

from ..cnn.cnn_base import ConvBlockSequential
from ..cnn.resnet import ResBlockSequential
from ..embedding.embedder import EmbedderSequential
from ..rearrange import RearrangeConvCentric
from ..transformer.transformer_base import TransformerBlock
from .distributions import DiagonalGaussianDistribution, DiracDistribution
from ..cnn.cnn_base import EmbedBlockSequential


class Quantization(nn.Module):
    """
    A quantization module for encoding and decoding data through various block types (conv, resnet, transformer).

    :param in_ch: Input channel size.
    :param z_ch: Latent space channel size for quantization.
    :param latent_ch: Latent space channel size for bottleneck processing.
    :param block_type: Block type for the quantization process ('conv', 'resnet', or 'transformer').
    :param spatial_dim_count: Number of spatial dimensions (2 or 3).
    """

    def __init__(
            self,
            in_ch: int,
            latent_ch: List[int],
            block_type: str,
            spatial_dim_count: int,
            blocks: List[str],
            embedders: List[EmbedderSequential] = None,
            dims: int = 2,
            distribution: str = "gaussian",
            **kwargs
    ):
        super().__init__()
        # Choose the block type based on provided configuration
        self.distribution: str = distribution
        if block_type == "ConvBlock":
            # Define convolutional block for quantization
            self.quant: nn.Module = RearrangeConvCentric(
                EmbedBlockSequential(
                    nn.GroupNorm(32, in_ch),  # Normalize input channels
                    ConvBlockSequential(in_ch, [(1 + (distribution == "gaussian")) * l_ch for l_ch in latent_ch], blocks, dims=dims, **kwargs)
                ), spatial_dim_count, dims=dims
            )
            # Define convolutional block for post-quantization decoding
            self.post_quant: nn.Module = RearrangeConvCentric(
                ConvBlockSequential(latent_ch[-1], latent_ch[::-1][1:] + [in_ch], blocks, dims=dims, **kwargs),
                spatial_dim_count, dims=dims
            )
        elif block_type == "ResNetBlock":
            # Define ResNet block for quantization
            self.quant = RearrangeConvCentric(
                EmbedBlockSequential(
                    ResBlockSequential(in_ch, [(1 + (distribution == "gaussian")) * l_ch for l_ch in latent_ch], blocks,
                                       embedders=embedders, dims=dims,
                                       **kwargs)
                ), spatial_dim_count, dims=dims
            )
            # Define ResNet block for post-quantization decoding
            self.post_quant = RearrangeConvCentric(
                ResBlockSequential(latent_ch[-1], latent_ch[::-1][1:] + [in_ch], blocks,
                                   embedders=embedders, dims=dims,
                                   **kwargs),
                spatial_dim_count, dims=dims
            )
        elif block_type == "TransformerBlock":
            self.quant = TransformerBlock(in_ch,
                                          [(1 + (distribution == "gaussian")) * l_ch for l_ch in latent_ch],
                                          blocks,
                                          spatial_dim_count=spatial_dim_count,
                                          embedders=embedders,
                                          **kwargs)
            self.post_quant = TransformerBlock(latent_ch[-1],
                                               latent_ch[::-1][1:] + [in_ch],
                                               blocks,
                                               spatial_dim_count=spatial_dim_count,
                                               embedders=embedders,
                                               **kwargs)
        else:
            raise ValueError(f"Unsupported block_type: {block_type}")

    def quantize(
        self,
        x: torch.Tensor,
        emb: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        *args: Any,
        **kwargs: Any
    ):
        """
        Encodes the input tensor x into a quantized latent space.

        :param x: Input tensor of shape ``(b, v, t, n, d, f)``.
        :param emb: Optional embedding tensor.
        :param mask: Optional mask tensor of shape ``(b, v, t, n, d, m)``.
        :param args: Additional positional arguments forwarded to the block.
        :param kwargs: Additional keyword arguments forwarded to the block.
        :return: Quantized tensor of shape ``(b, v, t, n, d, f')``.
        """
        return self.quant(x, emb=emb, mask=mask, *args, **kwargs)

    def post_quantize(
        self,
        x: torch.Tensor,
        emb: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        *args: Any,
        **kwargs: Any
    ):
        """
        Decodes the quantized tensor x back to the original space.

        :param x: Quantized tensor of shape ``(b, v, t, n, d, f')``.
        :param emb: Optional embedding tensor.
        :param mask: Optional mask tensor of shape ``(b, v, t, n, d, m)``.
        :param args: Additional positional arguments forwarded to the block.
        :param kwargs: Additional keyword arguments forwarded to the block.
        :return: Decoded tensor of shape ``(b, v, t, n, d, f)``.
        """
        return self.post_quant(x, emb=emb, mask=mask, *args, **kwargs)

    def get_distribution(self, x: torch.Tensor):
        """
        Encodes the input tensor x into a quantized latent space.

        :param x: Input tensor of shape ``(b, v, t, n, d, f')``.
        :return: Distribution instance for the tensor.
        """
        assert self.distribution == "gaussian" or self.distribution == "dirac"
        if self.distribution == "gaussian":
            return DiagonalGaussianDistribution(x)
        elif self.distribution == "dirac":
            return DiracDistribution(x)
