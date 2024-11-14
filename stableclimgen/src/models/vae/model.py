from typing import Optional, Union, List, Tuple

import torch
import torch.nn as nn

from .distributions import DiagonalGaussianDistribution
from .quantization import Quantization
from ...modules.embedding.patch import PatchEmbedder3D, ConvUnpatchify, LinearUnpatchify
from ...modules.rearrange import RearrangeConvCentric
from ...modules.cnn.conv import ConvBlockSequential
from ...modules.cnn.resnet import ResBlockSequential
from ...modules.transformer.transformer_base import TransformerBlock
from ...modules.utils import EmbedBlockSequential


class VAEBlockConfig:
    """
    Configuration class for defining the parameters of VAE blocks.

    :param depth: Number of layers in the block.
    :param block_type: Type of block (e.g., 'conv' or 'resnet').
    :param ch_mult: Channel multiplier for the block; can be an int or list of ints.
    :param blocks: List of block configurations specific to the block type.
    :param enc: Boolean indicating if this is an encoder block. Default is False.
    :param dec: Boolean indicating if this is a decoder block. Default is False.
    """

    def __init__(self, depth: int, block_type: str, ch_mult: Union[int, List[int]], blocks: List[str],
                 enc: bool = False, dec: bool = False):
        self.depth = depth
        self.block_type = block_type
        self.ch_mult = ch_mult
        self.blocks = blocks
        self.enc = enc
        self.dec = dec


class QuantConfig:
    """
    Configuration class for defining quantization parameters in the VAE model.

    :param z_ch: Number of latent channels in the quantized representation.
    :param latent_ch: Number of latent channels in the bottleneck.
    :param block_type: Block type used in quantization (e.g., 'conv' or 'resnet').
    """

    def __init__(self, z_ch: int, latent_ch: int, block_type: str):
        self.z_ch = z_ch
        self.latent_ch = latent_ch
        self.block_type = block_type


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) model using a configurable encoder-decoder architecture.

    :param init_in_ch: Initial input channel count.
    :param final_out_ch: Final output channel count.
    :param block_configs: List of VAEBlockConfig instances defining encoder/decoder blocks.
    :param quant_config: QuantConfig instance defining quantization configuration.
    :param model_channels: Number of model channels (default 64).
    :param embed_dim: Optional embedding dimension.
    :param patch_emb_type: Patch embedding type ("conv" or "linear").
    :param patch_emb_size: Patch embedding size (tuple of dimensions).
    :param patch_emb_kernel: Kernel size for patch embedding.
    :param concat_cond: Whether to concatenate conditional data.
    :param spatial_dim_count: Number of spatial dimensions (2 or 3).
    """

    def __init__(self,
                 init_in_ch: int,
                 final_out_ch: int,
                 block_configs: List[VAEBlockConfig],
                 quant_config: QuantConfig,
                 model_channels: int = 64,
                 embed_dim: Optional[int] = None,
                 patch_emb_type: str = "conv",
                 patch_emb_size: Union[Tuple[int, int], Tuple[int, int, int]] = (1, 1, 1),
                 patch_emb_kernel: Union[Tuple[int, int], Tuple[int, int, int]] = (1, 1, 1),
                 concat_cond: bool = False,
                 spatial_dim_count: int = 2):
        super().__init__()

        self.embed_dim = embed_dim
        in_ch = init_in_ch * (1 + concat_cond)  # Adjust input channels if concat_cond is used

        # Define input patch embedding layer
        self.input_patch_embedding = RearrangeConvCentric(PatchEmbedder3D(
            in_ch, int(model_channels), patch_emb_kernel, patch_emb_size
        ), spatial_dim_count)

        # Define encoder and decoder block lists
        enc_blocks, dec_blocks = [], []
        in_ch = bottleneck_ch = model_channels

        # Construct blocks based on configurations
        for block_conf in block_configs:
            for _ in range(block_conf.depth):
                # Calculate output channels based on channel multiplier
                out_ch = block_conf.ch_mult * model_channels if isinstance(block_conf.ch_mult, int) else [
                    ch_mult * model_channels for ch_mult in block_conf.ch_mult]

                # Select the block type (conv, resnet, or transformer)
                if block_conf.block_type == "conv":
                    block = RearrangeConvCentric(
                        ConvBlockSequential(in_ch, out_ch, block_conf.blocks), spatial_dim_count
                    )
                elif block_conf.block_type == "resnet":
                    block = RearrangeConvCentric(
                        ResBlockSequential(in_ch, out_ch, block_conf.blocks), spatial_dim_count
                    )
                else:
                    block = TransformerBlock(in_ch, out_ch, block_conf.blocks, spatial_dim_count=spatial_dim_count)

                in_ch = out_ch if isinstance(out_ch, int) else out_ch[-1]  # Update input channels for next layer

                # Append block to encoder or decoder based on configuration
                if block_conf.enc:
                    enc_blocks.append(block)
                    bottleneck_ch = in_ch
                else:
                    dec_blocks.append(block)

        # Define quantization block
        self.quantization = Quantization(in_ch=bottleneck_ch, z_ch=quant_config.z_ch, latent_ch=quant_config.latent_ch,
                                         block_type=quant_config.block_type, spatial_dim_count=spatial_dim_count)

        # Assemble encoder and decoder layers
        self.encoder = EmbedBlockSequential(*enc_blocks)
        self.decoder = EmbedBlockSequential(*dec_blocks)

        # Define output unpatchifying layer
        if patch_emb_type == "conv":
            self.out = RearrangeConvCentric(ConvUnpatchify(in_ch, final_out_ch), spatial_dim_count)
        else:
            self.out = LinearUnpatchify(in_ch, final_out_ch, patch_emb_size, embed_dim, spatial_dim_count)

    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        """
        Encodes input data to latent space, returning a posterior distribution.

        :param x: Input tensor.
        :return: DiagonalGaussianDistribution posterior distribution.
        """
        x = self.input_patch_embedding(x)
        h = self.encoder(x)
        moments = self.quantization.quantize(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes latent variable z back to data space.

        :param z: Latent tensor.
        :return: Decoded tensor in data space.
        """
        z = self.quantization.post_quantize(z)
        dec = self.decoder(z)
        return dec

    def forward(self, x: torch.Tensor, embeddings: Optional[torch.Tensor] = None,
                mask_data: Optional[torch.Tensor] = None,
                cond_data: Optional[torch.Tensor] = None, coords: Optional[torch.Tensor] = None,
                sample_posterior: bool = True) -> Tuple[torch.Tensor, DiagonalGaussianDistribution]:
        """
        Forward pass through the VAE, encoding and decoding the input.

        :param x: Input tensor.
        :param embeddings: Optional embeddings for conditional inputs.
        :param mask_data: Optional mask data for attention.
        :param cond_data: Optional conditional data for input.
        :param coords: Optional coordinates for spatial embedding.
        :param sample_posterior: Boolean flag to sample from posterior distribution.
        :return: Tuple of reconstructed tensor and posterior distribution.
        """
        # Define output shape for reconstruction
        out_shape = x.shape[1:-2]

        posterior = self.encode(x)
        z = posterior.sample() if sample_posterior else posterior.mode()
        dec = self.decode(z)
        return self.out(dec, out_shape=out_shape), posterior
