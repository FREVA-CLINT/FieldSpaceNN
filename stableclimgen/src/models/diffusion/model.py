from typing import List

import torch
import torch.nn as nn

from stableclimgen.src.modules.embedding.diffusion_step import DiffusionStepEmbedder
from stableclimgen.src.modules.embedding.patch import PatchEmbedder3D, LinearUnpatchify, ConvUnpatchify
from stableclimgen.src.modules.rearrange import RearrangeConvCentric
from stableclimgen.src.modules.utils import EmbedBlockSequential
from stableclimgen.src.modules.transformer.transformer_base import TransformerBlock
from stableclimgen.src.modules.cnn.resnet import ResBlock, ConvBlock


class DiffusionBlockConfig:
    def __init__(self, depth: int, block_type: str, ch_mult:float, sub_confs: dict, enc=False, dec=False):
        self.depth = depth
        self.block_type = block_type
        self.sub_confs = sub_confs
        self.enc = enc
        self.dec = dec
        self.ch_mult = ch_mult


class DiffusionGenerator(nn.Module):
    def __init__(
            self,
            init_in_ch,
            final_out_ch,
            model_channels=64,
            embed_dim=None,
            skip_connections=False,
            patch_emb_type="conv",
            patch_emb_size=(1, 1, 1),
            patch_emb_kernel=(1, 1, 1),
            block_configs: List[DiffusionBlockConfig] = None,
            concat_mask=False,
            concat_cond=False
    ):
        super().__init__()
        in_ch = init_in_ch * (1 + concat_mask + concat_cond)
        self.model_channels = model_channels
        self.skip_connections = skip_connections
        self.concat_mask = concat_mask
        self.concat_cond = concat_cond

        # define embeddings
        self.embed_dim = embed_dim
        self.diffusion_step_emb = DiffusionStepEmbedder(model_channels, embed_dim)

        self.input_patch_embedding = RearrangeConvCentric(PatchEmbedder3D(
            in_ch, model_channels * block_configs[0].ch_mult, patch_emb_kernel, patch_emb_size))

        enc_blocks, dec_blocks, prc_blocks = [], [], []

        in_ch = model_channels * block_configs[0].ch_mult
        in_block_ch = []

        # define input blocks
        for i, block_conf in enumerate(block_configs):
            for _ in range(block_conf.depth):
                out_ch = int(block_conf.ch_mult * model_channels)
                if self.skip_connections and block_conf.enc:
                    in_block_ch.append(in_ch)
                if self.skip_connections and block_conf.dec:
                    in_ch += in_block_ch.pop()
                if block_conf.block_type == "conv":
                    block = RearrangeConvCentric(
                        ConvBlock(in_ch, out_ch, *block_conf.sub_confs)
                    )
                elif block_conf.block_type == "resnet":
                    block = RearrangeConvCentric(
                        ResBlock(in_ch, out_ch, **block_conf.sub_confs)
                    )
                else:
                    block = TransformerBlock(in_ch, out_ch, **block_conf.sub_confs)
                in_ch = out_ch
                # separate for skip connections
                if block_conf.enc:
                    enc_blocks.append(block)
                elif block_conf.dec:
                    dec_blocks.append(block)
                else:
                    prc_blocks.append(block)
        self.encoder = EmbedBlockSequential(*enc_blocks)
        self.processor = EmbedBlockSequential(*prc_blocks)
        self.decoder = EmbedBlockSequential(*dec_blocks)

        if patch_emb_type == "conv":
            self.out = RearrangeConvCentric(ConvUnpatchify(out_ch, final_out_ch))
        else:
            self.out = LinearUnpatchify(out_ch, final_out_ch, patch_emb_size, embed_dim)

    def forward(self, x, diffusion_steps=None, mask=None, cond=None, coords=None):
        out_shape = x.shape[2:-1]

        if self.concat_mask:
            x = torch.cat([x, mask], dim=-1)
        if self.concat_cond:
            x = torch.cat([x, cond], dim=-1)

        h = self.input_patch_embedding(x)

        # embeddings
        emb = torch.zeros(*h.shape[:-1], self.embed_dim,device=h.device, layout=h.layout,dtype=h.dtype)

        emb.add_(self.diffusion_step_emb(diffusion_steps)[:, None, None, None, None, :])

        # remove if necessary
        mask = None

        hs = [h]
        # encoder
        for i, module in enumerate(self.encoder):
            h = module(h, emb[..., :h.shape[-4], :h.shape[-3], :h.shape[-2], :], mask, cond, coords)
            if self.skip_connections:
                hs.append(h)

        # processor
        for i, module in enumerate(self.processor):
            h = module(h, emb[..., :h.shape[-4], :h.shape[-3], :h.shape[-2], :], mask, cond, coords)

        # decoder
        for i, module in enumerate(self.decoder):
            if self.skip_connections:
                h = torch.cat([h, hs.pop()], dim=-1)
            h = module(h, emb[..., :h.shape[-4], :h.shape[-3], :h.shape[-2], :], None, cond, coords)

        # out layer
        h = self.out(h, emb[..., :h.shape[-4], :h.shape[-3], :h.shape[-2], :], mask, cond, coords, out_shape)
        return h
