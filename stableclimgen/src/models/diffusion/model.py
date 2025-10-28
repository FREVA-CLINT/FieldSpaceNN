from typing import List, Optional, Union, Dict, Tuple

import torch
import torch.nn as nn

from ..CNN.confs import CNNBlockConfig
from ...modules.embedding.patch import PatchEmbedderND, ConvUnpatchify
from ...modules.rearrange import RearrangeConvCentric
from ...modules.cnn.conv import ConvBlockSequential
from ...modules.cnn.resnet import ResBlockSequential
from ...modules.transformer.transformer_base import TransformerBlock
from ...utils.utils import EmbedBlockSequential


class DiffusionGenerator(nn.Module):
    """
    Diffusion Generator model class for generating data through a diffusion process.

    :param init_in_ch: Initial number of input channels.
    :param final_out_ch: Final number of output channels.
    :param model_channels: Number of base channels for the model. Default is 64.
    :param embed_dim: Embedding dimension for diffusion steps. Default is None.
    :param skip_connections: If True, skip connections are included. Default is False.
    :param patch_emb_type: Type of patch embedding ('ConvBlock' or other). Default is 'ConvBlock'.
    :param patch_emb_size: Size of the patch embedding. Default is (1, 1, 1).
    :param patch_emb_kernel: Kernel size for the patch embedding. Default is (1, 1, 1).
    :param block_configs: List of block configurations for each block in the model.
    :param concat_mask: If True, mask is concatenated to the input. Default is False.
    :param concat_cond: If True, conditioning data is concatenated to the input. Default is False.
    :param spatial_dim_count: Determines the number of the spatial dimensions
    """

    def __init__(
            self,
            init_in_ch: int,
            final_out_ch: int,
            block_configs: List[CNNBlockConfig],
            model_channels: int = 64,
            patch_emb_size: Tuple = (1, 1),
            patch_emb_kernel: Tuple = (1, 1),
            skip_connections: bool = False,
            concat_mask: bool = False,
            concat_cond: bool = False,
            spatial_dim_count: int = 2
    ):
        super().__init__()

        # Channel configurations
        in_ch = init_in_ch * (1 + concat_mask + concat_cond)
        self.model_channels = model_channels
        self.skip_connections = skip_connections
        self.concat_mask = concat_mask
        self.concat_cond = concat_cond
        self.dims = len(patch_emb_size)

        # Define input patch embedding layer
        self.input_patch_embedding = RearrangeConvCentric(PatchEmbedderND(
            in_ch, int(model_channels), patch_emb_kernel, patch_emb_size, self.dims
        ), spatial_dim_count, dims=self.dims)

        # Define encoder, processor, and decoder block lists
        enc_blocks, dec_blocks, prc_blocks = [], [], []
        in_ch = model_channels
        in_block_ch = []

        # Define layers based on block configurations
        for block_conf in block_configs:
            for d in range(block_conf.depth):
                # Calculate output channels based on channel multiplier
                out_ch = block_conf.ch_mult * model_channels if isinstance(block_conf.ch_mult, int) else [
                    ch_mult * model_channels for ch_mult in block_conf.ch_mult]

                if self.skip_connections and block_conf.enc:
                    in_block_ch.append(in_ch)
                if self.skip_connections and block_conf.dec:
                    in_ch += in_block_ch.pop()

                # Determine block type (ConvBlock, ResnetBlock, or TransformerBlock)
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

                in_ch = out_ch

                # Append block to encoder, decoder, or processor list
                if block_conf.enc:
                    enc_blocks.append(block)
                elif block_conf.dec:
                    dec_blocks.append(block)
                else:
                    prc_blocks.append(block)

        # Assign blocks to encoder, processor, and decoder layers
        self.encoder = EmbedBlockSequential(*enc_blocks)
        self.processor = EmbedBlockSequential(*prc_blocks)
        self.decoder = EmbedBlockSequential(*dec_blocks)

        self.out = RearrangeConvCentric(ConvUnpatchify(in_ch, final_out_ch, dims=self.dims), spatial_dim_count, dims=self.dims)

    def forward(
            self,
            x: torch.Tensor,
            emb: Optional[Dict] = None,
            mask: Optional[torch.Tensor] = None,
            cond: Optional[torch.Tensor] = None,
            **kwargs
    ) -> torch.Tensor:
        """
        Forward pass through the Diffusion Generator model.

        :param x: Input tensor.
        :param emb: Tensor representing different embeddings.
        :param mask: Mask tensor, used if `concat_mask` is True.
        :param cond: Conditioning tensor, used if `concat_cond` is True.
        :param coords:

        :return: Output tensor after processing through the model.
        """

        # Define the output shape for reconstruction
        out_shape = x.shape[-self.dims-1:-1]

        # Concatenate mask and conditioning if specified
        if self.concat_mask:
            x = torch.cat([x, mask.float()], dim=-1)
        if self.concat_cond:
            x = torch.cat([x, cond], dim=-1)

        # Initial patch embedding for the input tensor
        h = self.input_patch_embedding(x)

        # Remove mask if unnecessary
        mask = None

        # List to store intermediate states for skip connections
        hs = [h]

        # Encoder forward pass with optional skip connections
        for module in self.encoder:
            h = module(h, emb, mask)
            if self.skip_connections:
                hs.append(h)

        # Processor forward pass
        for module in self.processor:
            h = module(h, emb, mask)

        # Decoder forward pass with optional skip connections
        for module in self.decoder:
            if self.skip_connections:
                h = torch.cat([h, hs.pop()], dim=-1)
            h = module(h, emb, mask)

        # Output layer reconstruction with unpatchifying
        h = self.out(h, emb, mask, out_shape=out_shape)
        return h