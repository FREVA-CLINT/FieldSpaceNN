from typing import Optional, List, Tuple

import torch
import torch.nn as nn

from stableclimgen.src.modules.vae.quantization import Quantization
from ..cnn.confs import CNNBlockConfig, QuantConfig
from ...modules.cnn.cnn_base import ConvBlockSequential, PatchEmbedderND, ConvUnpatchify
from ...modules.cnn.resnet import ResBlockSequential
from ...modules.vae.distributions import AbstractDistribution
from ...modules.rearrange import RearrangeConvCentric
from ...modules.transformer.transformer_base import TransformerBlock
from ...modules.cnn.cnn_base import EmbedBlockSequential
from einops import rearrange


class CNN_VAE(nn.Module):
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
    :param var_to_feat: Whether to fold variables into the feature channel.
    """

    def __init__(
        self,
        init_in_ch: int,
        final_out_ch: int,
        block_configs: List[CNNBlockConfig],
        quant_config: QuantConfig,
        model_channels: int = 64,
        patch_emb_size: Tuple[int, ...] = (1, 1),
        patch_emb_kernel: Tuple[int, ...] = (1, 1),
        concat_cond: bool = False,
        spatial_dim_count: int = 2,
        var_to_feat: bool = False,
    ) -> None:
        """
        Initialize the CNN VAE encoder/decoder and quantization blocks.

        :param init_in_ch: Initial input channel count.
        :param final_out_ch: Final output channel count.
        :param block_configs: List of block configurations for encoder/decoder stacks.
        :param quant_config: Quantization configuration.
        :param model_channels: Base number of model channels.
        :param patch_emb_size: Patch embedding size per spatial dimension.
        :param patch_emb_kernel: Kernel size for patch embedding.
        :param concat_cond: Whether to concatenate conditional data.
        :param spatial_dim_count: Number of spatial dimensions (2 or 3).
        :param var_to_feat: Whether to fold variables into the feature channel.
        :return: None.
        """
        super().__init__()

        in_ch = init_in_ch * (1 + concat_cond)  # Adjust input channels if concat_cond is used

        self.dims: int = len(patch_emb_size)

        # Define input patch embedding layer
        self.input_patch_embedding: nn.Module = RearrangeConvCentric(PatchEmbedderND(
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
        self.quantization: Quantization = Quantization(
            in_ch=bottleneck_ch,
            latent_ch=quant_config.latent_ch,
            block_type=quant_config.block_type,
            **quant_config.sub_confs,
            spatial_dim_count=spatial_dim_count,
            dims=self.dims,
        )

        # Assemble encoder and decoder layers
        self.encoder: nn.Module = EmbedBlockSequential(*enc_blocks)
        self.decoder: nn.Module = EmbedBlockSequential(*dec_blocks)

        self.var_to_feat: bool = var_to_feat

        # Define output unpatchifying layer
        self.out: nn.Module = RearrangeConvCentric(
            ConvUnpatchify(in_ch, final_out_ch, dims=self.dims), spatial_dim_count, dims=self.dims
        )

    def encode(self, x: torch.Tensor) -> AbstractDistribution:
        """
        Encodes input data to latent space, returning a posterior distribution.

        :param x: Input tensor with shape ``(b, v, t, n, d, f)`` or a CNN-friendly view
            such as ``(b, c, h, w)`` depending on the model configuration.
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

        :param z: Latent tensor produced by the quantization block.
        :return: Decoded tensor in data space before unpatchifying.
        """
        z = self.quantization.post_quantize(z)
        dec = self.decoder(z)

        return dec

    def forward(
        self,
        x: torch.Tensor,
        embeddings: Optional[torch.Tensor] = None,
        mask_data: Optional[torch.Tensor] = None,
        cond_data: Optional[torch.Tensor] = None,
        coords: Optional[torch.Tensor] = None,
        sample_posterior: bool = True,
    ) -> Tuple[torch.Tensor, AbstractDistribution]:
        """
        Forward pass through the VAE, encoding and decoding the input.

        :param x: Input tensor with shape ``(b, v, t, n, d, f)`` or a CNN-friendly view.
        :param embeddings: Optional embeddings (currently unused by this module).
        :param mask_data: Optional mask data (currently unused by this module).
        :param cond_data: Optional conditional data (currently unused by this module).
        :param coords: Optional coordinates (currently unused by this module).
        :param sample_posterior: Boolean flag to sample from posterior distribution.
        :return: Tuple of reconstructed tensor and posterior distribution. The output
            matches the spatial shape of ``x`` and aligns with the base
            ``(b, v, t, n, d, f)`` convention when applicable.
        """
        # Define output shape for reconstruction
        if self.var_to_feat:
            v = x.shape[1]
            # Fold variables into the feature channel to match CNN expectations.
            x = rearrange(x, "b v t w h c -> b 1 t w h (c v)")

        out_shape = x.shape[-self.dims-1:-1]
        posterior = self.encode(x)
        z = posterior.sample() if sample_posterior else posterior.mode()
        dec = self.decode(z)

        output = self.out(dec, out_shape=out_shape)

        if self.var_to_feat:
            # Restore variable dimension from the feature channel.
            output = rearrange(output, "b 1 t w h (c v) -> b v t w h c", v=v)

        return output, posterior
