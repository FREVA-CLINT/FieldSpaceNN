import torch
import torch.nn as nn

from latengen.model.diffusion.attention import SelfAttention, TransformerBlock
from latengen.model.diffusion.modules import SpatioTemporalPatchEmbedding, ResBlock, Identity, \
    EmbedBlockSequential, FinalLayer, ConvBlock
from latengen.model.diffusion.nn import (
    linear,
    timestep_embedding
)
from latengen.model.diffusion.rearrange import RearrangeBatchCentric, RearrangeSpaceCentric, RearrangeTimeCentric, \
    RearrangeSpaceChannelCentric, RearrangeTimeChannelCentric


class DiffusionGenerator(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param layer_depth: number of residual blocks per downsample.
    :param spat_att_res: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param conv_factors: channel multiplier for each level of the UNet.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param n_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
            self,
            out_channels,
            img_sizes,
            layer_depth=2,
            model_channels=64,
            block_type="conv2d",
            spat_att_res=(),
            temp_att_res=(),
            time_window=None,
            seq_length=12,
            bottleneck_spat_att=True,
            bottleneck_temp_att=True,
            dropout=0.0,
            conv_factors=(1, 2, 4, 8),
            n_heads=1,
            n_head_channels=-1,
            updown=None,
            bottleneck=False,
            embed_dim=None,
            rel_position=None,
            skip_connections=False,
            patch_emb_size=(3, 3, 3),
            patch_emb_kernel=None,
            conditioning="channel",
            time_causal=False,
            max_rel_pos=None,
            disable_spat_att=False,
            disable_temp_att=False,
            concat_mask=False,
            channel_attention=False
    ):
        super().__init__()

        if max_rel_pos is None:
            max_rel_pos = seq_length

        self.image_size = img_sizes
        self.in_channels = in_channels = out_channels * (1 + (1 if conditioning == "channel" else 0) + concat_mask)
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.layer_depth = layer_depth
        self.dropout = dropout
        self.conv_factors = conv_factors
        self.skip_connections = skip_connections
        self.conditioning = conditioning
        self.concat_mask = concat_mask

        if patch_emb_kernel is None:
            patch_emb_kernel = patch_emb_size

        ch = input_ch = int(conv_factors[0] * model_channels)
        input_block_chans = [ch]
        curr_img_size = self.image_size
        # define embeddings
        if not embed_dim:
            embed_dim = model_channels * 4
        self.embed_dim = embed_dim

        self.time_embed = nn.Sequential(
            linear(model_channels, embed_dim),
            nn.SiLU(),
            linear(embed_dim, embed_dim),
        )

        def spat_attn_block(dim, img_size=None, cond_dim=None, embed=embed_dim):
            if disable_spat_att:
                return Identity()
            else:
                if channel_attention:
                    rearrange_fn = RearrangeSpaceChannelCentric
                    dim = embed = img_size[0] * img_size[1]
                else:
                    rearrange_fn = RearrangeSpaceCentric
                return rearrange_fn(
                    SelfAttention(
                        dim,
                        cond_dim,
                        dropout=dropout,
                        num_heads=n_heads,
                        num_head_channels=n_head_channels,
                        embed_dim=embed,
                        rel_position=rel_position,
                        window_size=img_size
                    ))

        def temp_attn_block(dim, t=None, cond_dim=None, embed=embed_dim):
            if disable_temp_att:
                return Identity()
            else:
                if channel_attention:
                    rearrange_fn = RearrangeTimeChannelCentric
                    dim = embed = t
                else:
                    rearrange_fn = RearrangeTimeCentric
                return rearrange_fn(
                    SelfAttention(
                        dim,
                        cond_dim,
                        dropout=dropout,
                        num_heads=n_heads,
                        num_head_channels=n_head_channels,
                        embed_dim=embed,
                        rel_position=rel_position,
                        window_size=t,
                        max_window_size=max_rel_pos
                    ))

        def block(in_ch, out_ch, block_type="conv2d", img_size=None, time_window=None, up=False, down=False):
            if block_type == "conv2d":
                return RearrangeBatchCentric(ConvBlock(in_ch, out_ch, up=up, down=down, img_size=img_size))
            if block_type == "resnet":
                return RearrangeBatchCentric(ResBlock(
                    in_ch,
                    embed_dim,
                    dropout=dropout,
                    out_channels=out_ch,
                    up=up, down=down, img_size=img_size))
            if block_type == "transformer":
                return TransformerBlock(in_ch, out_ch,
                                        spat_attn_block(in_ch, img_size),
                                        temp_attn_block(in_ch, time_window),
                                        spat_attn_block(in_ch, img_size, cond_dim=input_ch, embed=None) if conditioning == "cross_att" else None,
                                        temp_attn_block(in_ch, time_window, cond_dim=input_ch, embed=None) if conditioning == "cross_att" else None,
                                        time_window=time_window, embed_dim=embed_dim, up=up, down=down,
                                        time_causal=time_causal)

        if not time_window:
            time_window = len(conv_factors) * [seq_length]
        time_window = [t // patch_emb_size[0] for t in time_window]
        max_rel_pos = max_rel_pos // patch_emb_size[0]

        def patch_emb(in_ch, out_ch):
            return EmbedBlockSequential(SpatioTemporalPatchEmbedding(in_ch, out_ch, patch_emb_kernel, patch_emb_size))

        self.input_patch_embedding = patch_emb(in_channels, ch)
        if conditioning == "cross_att":
            self.cond_patch_embedding = patch_emb(in_channels, ch)
        elif conditioning == "embedding":
            self.cond_patch_embedding = patch_emb(in_channels, self.embed_dim)
            self.emb_activation = torch.nn.GELU()

        curr_img_size = (curr_img_size[0] // patch_emb_size[1] + (curr_img_size[0] % patch_emb_size[1] != 0),
                         curr_img_size[1] // patch_emb_size[2] + (curr_img_size[1] % patch_emb_size[2] != 0))
        img_sizes = [curr_img_size]

        self.in_blocks = nn.ModuleList([])
        # define input blocks
        for level, factor in enumerate(conv_factors):
            for _ in range(layer_depth):
                layers = [block(ch, int(factor * model_channels), block_type, curr_img_size, time_window[level])]
                ch = int(factor * model_channels)
                if curr_img_size[0] in spat_att_res:
                    layers.append(spat_attn_block(ch, curr_img_size))
                if curr_img_size[0] in temp_att_res:
                    layers.append(temp_attn_block(ch, time_window[level]))
                self.in_blocks.append(EmbedBlockSequential(*layers))
                input_block_chans.append(ch)
            if level != len(conv_factors) - 1 and updown is not None:
                self.in_blocks.append(EmbedBlockSequential(block(ch, ch, updown, curr_img_size, time_window[level], down=True)))
                input_block_chans.append(ch)
                curr_img_size = (curr_img_size[0] // 2 + (curr_img_size[0] % 2 != 0),
                                 curr_img_size[1] // 2 + (curr_img_size[1] % 2 != 0))
                img_sizes.append(curr_img_size)
        # define middle block
        mid_block = []
        if bottleneck:
            for _ in range(layer_depth):
                mid_block.append(block(ch, ch, block_type, curr_img_size, time_window[-1]))
                if bottleneck_spat_att:
                    mid_block.append(spat_attn_block(ch, curr_img_size))
                if bottleneck_temp_att:
                    mid_block.append(temp_attn_block(ch, time_window[-1]))
                if block_type == "resnet":
                    mid_block.append(block(ch, ch, block_type, curr_img_size, time_window[-1]))
        else:
            mid_block.append(nn.Identity())
        self.mid_block = EmbedBlockSequential(*mid_block)

        # define output blocks
        self.out_blocks = nn.ModuleList([])
        for level, factor in list(enumerate(conv_factors))[::-1]:
            for i in range(layer_depth + (1 if updown is not None else 0)):
                ich = input_block_chans.pop() * self.skip_connections
                layers = [block(ch + ich, int(factor * model_channels), block_type, curr_img_size, time_window[level])]
                ch = int(model_channels * factor)
                if curr_img_size[0] in spat_att_res:
                    layers.append(spat_attn_block(ch, curr_img_size))
                if curr_img_size[0] in temp_att_res:
                    layers.append(temp_attn_block(ch, time_window[level]))
                if level and i == layer_depth and updown is not None:
                    curr_img_size = img_sizes[level - 1]
                    layers.append(block(ch, ch, updown, curr_img_size, time_window[level], up=True))
                self.out_blocks.append(EmbedBlockSequential(*layers))

        self.out = EmbedBlockSequential(FinalLayer(
            input_ch, out_channels, patch_emb_size, embed_dim, curr_img_size, time_window[0])
        )

    def forward(self, x, diffusion_steps=None, mask=None, cond_input=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param diffusion_steps: a 1-D batch of timesteps.
        :param mask: an [N x C x ...] Tensor of inputs.
        :param cond_input: an [N x C x ...] Tensor of inputs.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []

        if self.conditioning == "channel" and torch.is_tensor(cond_input):
            x = torch.cat([x, cond_input], dim=1)
            cond_input = None

        if self.concat_mask:
            x = torch.cat([x, mask], dim=1)
            if torch.is_tensor(cond_input):
                cond_input = torch.cat([cond_input, mask], dim=1)

        h = self.input_patch_embedding(x)
        if torch.is_tensor(cond_input):
            cond_input = self.cond_patch_embedding(cond_input)

        # embeddings
        emb = torch.zeros(
            h.shape[0], self.embed_dim, *h.shape[-3:],
            device=h.device, layout=h.layout,
            dtype=h.dtype
        )

        emb.add_(self.time_embed(timestep_embedding(diffusion_steps, self.model_channels))[..., None, None, None])
        if self.conditioning == "embedding":
            emb.add_(cond_input)
            emb = self.emb_activation(emb)
            cond_input = None

        # input blocks
        for i, module in enumerate(self.in_blocks):
            h = module(h, emb[..., :h.shape[-3], :h.shape[-2], :h.shape[-1]], None, cond_input)
            if self.skip_connections:
                hs.append(h)
        # middle block
        h = self.mid_block(h, emb[..., :h.shape[-3], :h.shape[-2], :h.shape[-1]], None, cond_input)
        # output blocks
        for i, module in enumerate(self.out_blocks):
            if self.skip_connections:
                h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb[..., :h.shape[-3], :h.shape[-2], :h.shape[-1]], None, cond_input)
        # final layer
        for i, module in enumerate(self.out):
            h = module(h, emb[..., :h.shape[-3], :h.shape[-2], :h.shape[-1]], None, cond_input)
        return h
