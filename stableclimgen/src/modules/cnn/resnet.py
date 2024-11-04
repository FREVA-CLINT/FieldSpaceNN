import torch
import torch.nn as nn

from stableclimgen.src.modules.utils import EmbedBlock


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param in_channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, in_channels, out_channels, upsample, use_conv=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.use_conv = use_conv
        self.upsample = upsample
        if use_conv:
            self.conv = nn.Conv3d(self.in_channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.in_channels
        x = self.upsample(x)
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param in_channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, in_channels, out_channels, use_conv=True):
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

    def forward(self, x):
        assert x.shape[1] == self.in_channels
        return self.op(x)


class ConvBlock(EmbedBlock):
    def __init__(self, in_channels, out_channels, block_type):
        super().__init__()
        if block_type == "up":
            self.conv = Upsample(in_channels, out_channels, nn.Upsample(scale_factor=2, mode="nearest"))
        elif block_type == "down":
            self.conv = Downsample(in_channels, out_channels=out_channels)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)

    def forward(self, x, emb=None, mask=None, cond=None, coords=None, **kwargs):
        return self.conv(x)


class ResBlock(EmbedBlock):
    """
    A residual block that can optionally change the number of channels.

    :param in_ch: the number of input channels.
    :param embed_dim: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_ch: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
            self,
            in_ch,
            out_ch,
            embed_dim,
            block_type,
            dropout=0.,
            use_conv=False,
            use_scale_shift_norm=False
    ):
        super().__init__()
        self.in_ch = in_ch
        self.emb_channels = embed_dim
        self.dropout = dropout
        self.out_channels = out_ch or in_ch
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            nn.SiLU(),
            nn.Conv3d(in_ch, self.out_channels, 3, padding=1),
        )

        self.updown = (block_type == "up" or block_type == "down")

        if block_type=="up":
            self.h_upd = Upsample(in_ch, in_ch, nn.Upsample(scale_factor=(1, 2, 2), mode="nearest"), True)
            self.x_upd = Upsample(in_ch, in_ch, nn.Upsample(scale_factor=(1, 2, 2), mode="nearest"), True)
        elif block_type=="down":
            self.h_upd = Downsample(in_ch, in_ch,True)
            self.x_upd = Downsample(in_ch, in_ch,True)
        else:
            self.h_upd = self.x_upd = nn.Identity()

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

    def forward(self, x, emb=None, mask=None, cond=None, coords=None, **kwargs):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb = emb[:, :, 0, 0, 0]
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        skip = self.skip_connection(x)
        return skip + h
