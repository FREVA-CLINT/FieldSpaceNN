import torch
import torch.nn as nn
from typing import Optional, Union, List, Tuple

from stableclimgen.src.utils.helpers import check_value
from stableclimgen.src.utils.utils import EmbedBlockSequential


def conv_nd(in_channels: int,
            out_channels: Optional[int],
            kernel_size: int | Tuple[int, int, int],
            stride: int | Tuple[int, int, int] = 1,
            padding: int | Tuple[int, int, int] = 0,
            dims: int = 3):
    assert dims == 1 or dims == 2 or dims == 3
    return getattr(nn, f"Conv{dims}d")(in_channels, out_channels or in_channels, kernel_size, stride, padding)

def down_nd(stride, method="Avg", dims=2):
    assert dims == 1 or dims == 2 or dims == 3
    assert method == "Avg" or method == "Max"
    return getattr(nn, f"{method}Pool{dims}d")(stride, stride)


def zero_module(module: nn.Module) -> nn.Module:
    """
    Zero out the parameters of a module and return it.

    :param module: A PyTorch module whose parameters are to be zeroed out.
    :return: The same module with all parameters set to zero.
    """
    for p in module.parameters():
        p.detach().zero_()  # Detach and zero out the parameters to prevent gradient updates.
    return module


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param in_channels: Channels in the input tensor.
    :param out_channels: Channels in the output tensor; if None, defaults to in_channels.
    :param kernel_size: Kernel size for the convolution.
    :param padding: Padding size for the convolution.
    :param upsample: Custom upsampling module. If None, nearest-neighbor upsampling is used.
    :param use_conv: Boolean indicating if a convolution layer is applied after upsampling.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: Optional[int],
                 kernel_size: int | Tuple,
                 padding: int | Tuple,
                 upsample: Optional[nn.Module] = None,
                 use_conv: bool = True,
                 dims: int = 2):
        super().__init__()
        self.use_conv = use_conv
        # Define upsampling operation, using nearest-neighbor if none is provided
        self.upsample = upsample or nn.Upsample(scale_factor=tuple(x + 1 for x in padding), mode="nearest")
        # Define convolution layer if use_conv is True
        if use_conv:
            self.conv = conv_nd(in_channels, out_channels, kernel_size, 1, padding, dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Upsample layer.

        :param x: Input tensor of shape (batch, channels, depth, height, width).
        :return: Upsampled tensor, with convolution applied if use_conv is True.
        """
        x = self.upsample(x)  # Perform upsampling
        if self.use_conv:
            x = self.conv(x)  # Apply convolution if enabled
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param in_channels: Channels in the input tensor.
    :param out_channels: Channels in the output tensor; if None, defaults to in_channels.
    :param kernel_size: Kernel size for the convolution or pooling operation.
    :param padding: Padding size for the convolution.
    :param use_conv: Boolean indicating if a convolution layer is applied for downsampling.
    """

    def __init__(self, in_channels: int,
                 out_channels: Optional[int],
                 kernel_size: int | Tuple,
                 padding: int | Tuple,
                 use_conv: bool = True,
                 dims: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.use_conv = use_conv
        stride = 2  # Define stride for downsampling in height and width dimensions
        # Select convolution or average pooling for downsampling
        if use_conv:
            self.op = conv_nd(self.in_channels, self.out_channels, kernel_size, stride, padding, dims)
        else:
            self.op = nn.AvgPool3d(kernel_size=stride, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Downsample layer.

        :param x: Input tensor of shape (batch, channels, depth, height, width).
        :return: Downsampled tensor, with convolution applied if use_conv is True.
        """
        assert x.shape[1] == self.in_channels, "Input channel mismatch"
        return self.op(x)  # Apply the selected downsampling operation


class ConvBlockSequential(nn.Module):
    """
    A sequential convolutional block for upsampling, downsampling, or identity transformations.

    :param in_ch: Number of input channels.
    :param out_ch: List of output channels for each block in the sequence.
    :param blocks: List indicating the types of blocks ("up", "down", or identity).
    :param kernel_size: Kernel size for each block in the sequence, either a single tuple or a list of tuples.
    """

    def __init__(self, in_ch: int, out_ch: List[int], blocks: Union[str, List[str]],
                 kernel_size: int | List[int] | List[List[int]] = 3, dims=2, **kwargs):
        super().__init__()
        if isinstance(blocks, str):
            blocks = [blocks]
        kernel_size = check_value(kernel_size, len(blocks))
        conv_blocks = []
        for i, block in enumerate(blocks):
            kernel_size[i] = check_value(kernel_size[i], dims)
            # Calculate padding for each dimension based on kernel size
            padding = [kernel_size[i][j] // 2 for j in range(len(kernel_size[i]))]
            out_channels = out_ch if isinstance(out_ch, int) else out_ch[i]  # Determine output channels
            if block == "up":
                # Add an upsampling block
                conv_blocks.append(Upsample(in_ch, out_channels, kernel_size[i], padding, dims=dims))
            elif block == "down":
                # Add a downsampling block
                conv_blocks.append(Downsample(in_ch, out_channels, kernel_size[i], padding, dims=dims))
            else:
                # Add an identity convolutional block
                conv_blocks.append(conv_nd(in_ch, out_channels, kernel_size[i], padding=padding, dims=dims))
            in_ch = out_channels  # Update in_ch for the next layer

        # Create a sequential container for the convolutional blocks
        self.conv_blocks = EmbedBlockSequential(*conv_blocks)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass through the sequence of convolutional blocks.

        :param x: Input tensor.
        :return: Output tensor after applying the sequence of blocks.
        """
        return self.conv_blocks(x)  # Apply each block in sequence to the input tensor
