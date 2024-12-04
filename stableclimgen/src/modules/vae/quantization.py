from typing import Optional, List, Dict
import torch
import torch.nn as nn

from ..rearrange import RearrangeConvCentric
from ..cnn.conv import ConvBlockSequential
from ..cnn.resnet import ResBlockSequential
from ..transformer.transformer_base import TransformerBlock
from ..utils import EmbedBlockSequential


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
            embedder_names: List[List[str]] = None,
            embed_confs: Dict = None,
            embed_mode: str = "sum",):
        super().__init__()
        # Choose the block type based on provided configuration
        if block_type == "conv":
            # Define convolutional block for quantization
            self.quant = RearrangeConvCentric(
                EmbedBlockSequential(
                    nn.GroupNorm(32, in_ch),  # Normalize input channels
                    ConvBlockSequential(in_ch, [2 * l_ch for l_ch in latent_ch], blocks, [(1, 3, 3), (1, 1, 1)])
                ), spatial_dim_count
            )
            # Define convolutional block for post-quantization decoding
            self.post_quant = RearrangeConvCentric(
                ConvBlockSequential(latent_ch[-1], latent_ch[::-1][1:] + [in_ch], blocks, [(1, 1, 1), (1, 3, 3)]),
                spatial_dim_count
            )
        elif block_type == "resnet":
            # Define ResNet block for quantization
            self.quant = RearrangeConvCentric(
                EmbedBlockSequential(
                    nn.GroupNorm(32, in_ch),  # Normalize input channels
                    ResBlockSequential(in_ch, [2 * l_ch for l_ch in latent_ch], blocks, [(1, 3, 3), (1, 1, 1)],
                                       embedder_names=embedder_names, embed_confs=embed_confs, embed_mode=embed_mode)
                ), spatial_dim_count
            )
            # Define ResNet block for post-quantization decoding
            self.post_quant = RearrangeConvCentric(
                ResBlockSequential(latent_ch[-1], latent_ch[::-1][1:] + [in_ch], blocks, [(1, 1, 1), (1, 3, 3)],
                                   embedder_names=embedder_names, embed_confs=embed_confs, embed_mode=embed_mode),
                spatial_dim_count
            )
        else:
            # Use Transformer block for quantization and post-quantization
            self.quant = TransformerBlock(in_ch, [2 * l_ch for l_ch in latent_ch], blocks, spatial_dim_count=spatial_dim_count,
                                          embedder_names=embedder_names, embed_confs=embed_confs, embed_mode=embed_mode)
            self.post_quant = TransformerBlock(latent_ch[-1], latent_ch[::-1][1:] + [in_ch], blocks, spatial_dim_count=spatial_dim_count,
                                               embedder_names=embedder_names, embed_confs=embed_confs, embed_mode=embed_mode)

    def quantize(self, x: torch.Tensor, emb: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None,
                 cond: Optional[torch.Tensor] = None, coords: Optional[torch.Tensor] = None, *args, **kwargs) -> torch.Tensor:
        """
        Encodes the input tensor x into a quantized latent space.

        :param x: Input tensor.
        :param emb: Optional embedding tensor.
        :param mask: Optional mask tensor.
        :param cond: Optional conditioning tensor.
        :param coords: Optional coordinates tensor.
        :return: Quantized tensor.
        """
        return self.quant(x, emb, mask, cond, coords, *args, **kwargs)

    def post_quantize(self, x: torch.Tensor, emb: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None,
                      cond: Optional[torch.Tensor] = None, coords: Optional[torch.Tensor] = None, *args, **kwargs) -> torch.Tensor:
        """
        Decodes the quantized tensor x back to the original space.

        :param x: Quantized tensor.
        :param emb: Optional embedding tensor.
        :param mask: Optional mask tensor.
        :param cond: Optional conditioning tensor.
        :param coords: Optional coordinates tensor.
        :return: Decoded tensor.
        """
        return self.post_quant(x, emb, mask, cond, coords, *args, **kwargs)