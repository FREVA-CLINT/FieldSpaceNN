from typing import List, Optional, Union, Dict

import torch
import torch.nn as nn

# Import necessary modules for the diffusion generator architecture
from ...modules.embedding.patch import PatchEmbedderND, LinearUnpatchify, ConvUnpatchify
from ...modules.rearrange import RearrangeConvCentric
from ...modules.cnn.conv import ConvBlockSequential
from ...modules.cnn.resnet import ResBlockSequential
from ...modules.transformer.transformer_base import TransformerBlock
from ...utils.utils import EmbedBlockSequential
from ...utils.helpers import check_value


class DiffusionBlockConfig:
    """
    Configuration class for defining diffusion blocks in the model.

    :param depth: Number of layers in the block.
    :param block_type: Type of block (e.g., 'conv', 'resnet').
    :param ch_mult: Channel multiplier for the block.
    :param sub_confs: Sub-configuration details specific to the block type.
    :param enc: Whether the block is an encoder block. Default is False.
    :param dec: Whether the block is a decoder block. Default is False.
    """

    def __init__(self, depth: int, block_type: str, ch_mult: float, sub_confs: dict, enc: bool = False,
                 dec: bool = False, embedders: List[List[str]] = None):
        self.depth = depth
        self.block_type = block_type
        self.sub_confs = sub_confs
        self.enc = enc
        self.dec = dec
        self.ch_mult = ch_mult
        self.embedders = embedders


class PatchEmbConfig:
    """
    Configuration class for defining diffusion blocks in the model.

    :param depth: Number of layers in the block.
    :param block_type: Type of block (e.g., 'conv', 'resnet').
    :param ch_mult: Channel multiplier for the block.
    :param sub_confs: Sub-configuration details specific to the block type.
    :param enc: Whether the block is an encoder block. Default is False.
    :param dec: Whether the block is a decoder block. Default is False.
    """

    def __init__(self,
                 block_type: str = "conv",
                 patch_emb_size: tuple[int, int] | tuple[int, int, int] = (1, 1, 1),
                 patch_emb_kernel: tuple[int, int] | tuple[int, int, int] = (1, 1, 1),
                 sub_confs=None):
        self.block_type = block_type
        self.patch_emb_size = patch_emb_size
        self.patch_emb_kernel = patch_emb_kernel
        self.sub_confs = sub_confs


class DiffusionGenerator(nn.Module):
    """
    Diffusion Generator model class for generating data through a diffusion process.

    :param init_in_ch: Initial number of input channels.
    :param final_out_ch: Final number of output channels.
    :param model_channels: Number of base channels for the model. Default is 64.
    :param embed_dim: Embedding dimension for diffusion steps. Default is None.
    :param skip_connections: If True, skip connections are included. Default is False.
    :param patch_emb_type: Type of patch embedding ('conv' or other). Default is 'conv'.
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
            block_configs: List[DiffusionBlockConfig],
            patch_emb_config: PatchEmbConfig,
            model_channels: int = 64,
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
        self.dims = len(patch_emb_config.patch_emb_size)

        # Define input patch embedding
        self.input_patch_embedding = RearrangeConvCentric(PatchEmbedderND(
            in_ch, int(model_channels * block_configs[0].ch_mult), patch_emb_config.patch_emb_kernel, patch_emb_config.patch_emb_size, dims=self.dims
        ), spatial_dim_count, dims=self.dims)

        # Define encoder, processor, and decoder block lists
        enc_blocks, dec_blocks, prc_blocks = [], [], []
        in_ch = model_channels * block_configs[0].ch_mult
        in_block_ch = []

        # Define layers based on block configurations
        for block_conf in block_configs:
            for d in range(block_conf.depth):
                out_ch = int(block_conf.ch_mult * model_channels)
                if self.skip_connections and block_conf.enc:
                    in_block_ch.append(in_ch)
                if self.skip_connections and block_conf.dec:
                    in_ch += in_block_ch.pop()

                # Determine block type (conv, resnet, or transformer)
                if block_conf.block_type == "conv":
                    block = RearrangeConvCentric(
                        ConvBlockSequential(in_ch, out_ch, **block_conf.sub_confs, dims=self.dims), spatial_dim_count, dims=self.dims
                    )
                elif block_conf.block_type == "resnet":
                    block = RearrangeConvCentric(
                        ResBlockSequential(in_ch, out_ch, **block_conf.sub_confs, dims=self.dims), spatial_dim_count, dims=self.dims
                    )
                else:
                    block = TransformerBlock(in_ch, out_ch, **block_conf.sub_confs, spatial_dim_count=spatial_dim_count)

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

        # Define output unpatchifying layer
        if patch_emb_config.block_type == "conv":
            self.out = RearrangeConvCentric(ConvUnpatchify(out_ch, final_out_ch,
                                                           kernel_size=patch_emb_config.patch_emb_kernel, dims=self.dims),
                                            spatial_dim_count, dims=self.dims)
        else:
            self.out = LinearUnpatchify(out_ch, final_out_ch, patch_emb_config.patch_emb_size,
                                        **patch_emb_config.sub_confs, spatial_dim_count=spatial_dim_count)

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
        out_shape = x.shape[-2-self.dims:-2]

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