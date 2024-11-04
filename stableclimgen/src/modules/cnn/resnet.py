import torch
import torch.nn as nn
from typing import Optional
from stableclimgen.src.modules.utils import EmbedBlock


def zero_module(module: nn.Module) -> nn.Module:
    """
    Zero out the parameters of a module and return it.

    :param module: A PyTorch module whose parameters are to be zeroed out.
    :return: The same module with all parameters set to zero.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param in_channels: Channels in the inputs and outputs.
    :param out_channels: Channels in the output; if None, defaults to in_channels.
    :param upsample: The upsampling operation to apply.
    :param use_conv: A boolean determining if a convolution is applied.
    """

    def __init__(self, in_channels: int, out_channels: Optional[int], upsample: nn.Module, use_conv: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.use_conv = use_conv
        self.upsample = upsample
        if use_conv:
            self.conv = nn.Conv3d(self.in_channels, self.out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Upsample layer.

        :param x: Input tensor of shape (batch, channels, depth, height, width).
        :return: Upsampled tensor with or without convolution applied.
        """
        assert x.shape[1] == self.in_channels
        x = self.upsample(x)
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param in_channels: Channels in the inputs and outputs.
    :param out_channels: Channels in the output; if None, defaults to in_channels.
    :param use_conv: A boolean determining if a convolution is applied.
    """

    def __init__(self, in_channels: int, out_channels: Optional[int], use_conv: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.use_conv = use_conv
        stride = (1, 2, 2)
        if use_conv:
            self.op = nn.Conv3d(self.in_channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            assert self.in_channels == self.out_channels
            self.op = nn.AvgPool3d(kernel_size=stride, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Downsample layer.

        :param x: Input tensor of shape (batch, channels, depth, height, width).
        :return: Downsampled tensor with or without convolution applied.
        """
        assert x.shape[1] == self.in_channels
        return self.op(x)


class ConvBlock(EmbedBlock):
    """
    Convolutional block for upsampling or downsampling.

    :param in_channels: Number of input channels.
    :param out_channels: Number of output channels.
    :param block_type: Type of block, either "up" or "down".
    """

    def __init__(self, in_channels: int, out_channels: int, block_type: str):
        super().__init__()
        if block_type == "up":
            self.conv = Upsample(in_channels, out_channels, nn.Upsample(scale_factor=2, mode="nearest"))
        elif block_type == "down":
            self.conv = Downsample(in_channels, out_channels=out_channels)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)

    def forward(self, x: torch.Tensor, emb: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None,
                cond: Optional[torch.Tensor] = None, coords: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        Forward pass for the ConvBlock.

        :param x: Input tensor.
        :param emb: Optional embedding tensor.
        :param mask: Optional mask tensor.
        :param cond: Optional conditioning tensor.
        :param coords: Optional coordinates tensor.
        :return: Output tensor after applying the convolutional block.
        """
        return self.conv(x)


class ResBlock(EmbedBlock):
    """
    A residual block that can optionally change the number of channels.

    :param in_ch: Number of input channels.
    :param out_ch: Number of output channels (optional).
    :param embed_dim: Number of embedding channels.
    :param block_type: Type of block ("up", "down", or identity).
    :param dropout: Dropout rate (optional).
    :param use_conv: Whether to use a spatial convolution in the skip connection.
    :param use_scale_shift_norm: Boolean to determine if scale-shift normalization is used.
    """

    def __init__(self,
                 in_ch: int,
                 out_ch: Optional[int],
                 embed_dim: int,
                 block_type: str,
                 dropout: float = 0.0,
                 use_conv: bool = False,
                 use_scale_shift_norm: bool = False):
        super().__init__()
        self.in_ch = in_ch
        self.emb_channels = embed_dim
        self.dropout = dropout
        self.out_channels = out_ch or in_ch
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm

        # Define input layers with GroupNorm, activation, and convolution
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            nn.SiLU(),
            nn.Conv3d(in_ch, self.out_channels, 3, padding=1),
        )

        self.updown = (block_type == "up" or block_type == "down")

        if block_type == "up":
            self.h_upd = Upsample(in_ch, in_ch, nn.Upsample(scale_factor=(1, 2, 2), mode="nearest"), True)
            self.x_upd = Upsample(in_ch, in_ch, nn.Upsample(scale_factor=(1, 2, 2), mode="nearest"), True)
        elif block_type == "down":
            self.h_upd = Downsample(in_ch, in_ch, True)
            self.x_upd = Downsample(in_ch, in_ch, True)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        # Define embedding and output layers
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                embed_dim,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == in_ch:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv3d(in_ch, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = nn.Conv3d(in_ch, self.out_channels, 1)

    def forward(self, x: torch.Tensor, emb: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None,
                cond: Optional[torch.Tensor] = None, coords: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        Forward pass for the ResBlock.

        :param x: Input tensor.
        :param emb: Optional embedding tensor.
        :param mask: Optional mask tensor.
        :param cond: Optional conditioning tensor.
        :param coords: Optional coordinates tensor.
        :return: Output tensor after applying the residual block with skip connection.
        """
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb = emb[:, :, 0, 0, 0]  # Assume 5D tensor; extract relevant embedding part
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        # Apply scale-shift normalization if configured
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)

        # Add skip connection and return
        skip = self.skip_connection(x)
        return skip + h
