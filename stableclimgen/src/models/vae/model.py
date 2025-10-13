from typing import Optional, List, Tuple

import torch
import torch.nn as nn

from stableclimgen.src.modules.vae.quantization import Quantization
from ...modules.cnn.conv import ConvBlockSequential
from ...modules.cnn.resnet import ResBlockSequential
from ...modules.distributions.distributions import DiagonalGaussianDistribution, AbstractDistribution
from ...modules.embedding.patch import PatchEmbedderND, ConvUnpatchify
from ...modules.rearrange import RearrangeConvCentric
from ...modules.transformer.transformer_base import TransformerBlock
from ...utils.utils import EmbedBlockSequential
from einops import rearrange


class VAEBlockConfig:
    """
    Configuration class for defining the parameters of VAE blocks.

    :param depth: Number of layers in the block.
    :param block_type: Type of block (e.g., 'ConvBlock' or 'ResnetBlock').
    :param ch_mult: Channel multiplier for the block; can be an int or list of ints.
    :param blocks: List of block configurations specific to the block type.
    :param enc: Boolean indicating if this is an encoder block. Default is False.
    :param dec: Boolean indicating if this is a decoder block. Default is False.
    """

    def __init__(self, depth: int, block_type: str, ch_mult: int | List[int], sub_confs: dict, enc: bool = False,
                 dec: bool = False):
        self.depth = depth
        self.block_type = block_type
        self.sub_confs = sub_confs
        self.enc = enc
        self.dec = dec
        self.ch_mult = ch_mult


class QuantConfig:
    """
    Configuration class for defining quantization parameters in the VAE model.

    :param z_ch: Number of latent channels in the quantized representation.
    :param latent_ch: Number of latent channels in the bottleneck.
    :param block_type: Block type used in quantization (e.g., 'ConvBlock' or 'ResnetBlock').
    """

    def __init__(self, latent_ch: List[int], block_type: str, sub_confs: dict):
        self.latent_ch = latent_ch
        self.block_type = block_type
        self.sub_confs = sub_confs


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) model using a configurable encoder-decoder architecture.

    :param init_in_ch: Initial input channel count.
    :param final_out_ch: Final output channel count.
    :param block_configs: List of VAEBlockConfig instances defining encoder/decoder blocks.
    :param quant_config: QuantConfig instance defining quantization configuration.
    :param model_channels: Number of model channels (default 64).
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
                 patch_emb_size: Tuple = (1, 1),
                 patch_emb_kernel: Tuple = (1, 1),
                 concat_cond: bool = False,
                 spatial_dim_count: int = 2,
                 var_to_feat=False):
        super().__init__()

        in_ch = init_in_ch * (1 + concat_cond)  # Adjust input channels if concat_cond is used

        self.dims = len(patch_emb_size)

        # Define input patch embedding layer
        self.input_patch_embedding = RearrangeConvCentric(PatchEmbedderND(
            in_ch, int(model_channels), patch_emb_kernel, patch_emb_size, self.dims
        ), spatial_dim_count, dims=self.dims)

        # Define encoder and decoder block lists
        enc_blocks, dec_blocks = [], []
        in_ch = bottleneck_ch = model_channels

        # Construct blocks based on configurations
        for block_conf in block_configs:
            for _ in range(block_conf.depth):
                # Calculate output channels based on channel multiplier
                out_ch = block_conf.ch_mult * model_channels if isinstance(block_conf.ch_mult, int) else [
                    ch_mult * model_channels for ch_mult in block_conf.ch_mult]

                # Select the block type (ConvBlock, ResnetBlock, or TransformerBlock)
                if block_conf.block_type == "ConvBlock":
                    block = RearrangeConvCentric(
                        ConvBlockSequential(in_ch, out_ch, **block_conf.sub_confs, dims=self.dims), spatial_dim_count, dims=self.dims
                    )
                elif block_conf.block_type == "ResnetBlock":
                    block = RearrangeConvCentric(
                        ResBlockSequential(in_ch, out_ch, **block_conf.sub_confs, dims=self.dims), spatial_dim_count, dims=self.dims
                    )
                elif block_conf.block_type == "TransformerBlock":
                    block = TransformerBlock(in_ch, out_ch, **block_conf.sub_confs, spatial_dim_count=spatial_dim_count)
                else:
                    raise NotImplementedError

                in_ch = out_ch if isinstance(out_ch, int) else out_ch[-1]  # Update input channels for next layer

                # Append block to encoder or decoder based on configuration
                if block_conf.enc:
                    enc_blocks.append(block)
                    bottleneck_ch = in_ch
                else:
                    dec_blocks.append(block)

        # Define quantization block
        self.quantization = Quantization(in_ch=bottleneck_ch, latent_ch=quant_config.latent_ch, block_type=quant_config.block_type,
                                         **quant_config.sub_confs, spatial_dim_count=spatial_dim_count, dims=self.dims)

        # Assemble encoder and decoder layers
        self.encoder = EmbedBlockSequential(*enc_blocks)
        self.decoder = EmbedBlockSequential(*dec_blocks)

        self.var_to_feat = var_to_feat

        # Define output unpatchifying layer
        self.out = RearrangeConvCentric(ConvUnpatchify(in_ch, final_out_ch, dims=self.dims), spatial_dim_count, dims=self.dims)

    def encode(self, x: torch.Tensor) -> AbstractDistribution:
        """
        Encodes input data to latent space, returning a posterior distribution.

        :param x: Input tensor.
        :return: AbstractDistribution posterior distribution.
        """
        x = self.input_patch_embedding(x)
        h = self.encoder(x)
        moments = self.quantization.quantize(h)
        posterior = self.quantization.get_distribution(moments)
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
                sample_posterior: bool = True) -> Tuple[torch.Tensor, AbstractDistribution]:
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
        out_shape = x.shape[-self.dims-1:-1]

        if self.var_to_feat:
            v = x.shape[1]
            x = rearrange(x, "b v t w h c -> b 1 t w h (c v)")

        posterior = self.encode(x)
        z = posterior.sample() if sample_posterior else posterior.mode()
        dec = self.decode(z)

        output = self.out(dec, out_shape=out_shape)

        if self.var_to_feat:
            output = rearrange(output, "b 1 t w h (c v) -> b v t w h c", v=v)

        return output, posterior
