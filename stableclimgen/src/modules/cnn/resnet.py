import torch
import torch.nn as nn
from typing import Optional, Union, List, Tuple, Dict

from stableclimgen.src.modules.cnn.conv import Upsample, Downsample
from stableclimgen.src.modules.utils import EmbedBlock, EmbedBlockSequential
from stableclimgen.src.utils.helpers import expand_tensor_x_to_y, expand_tensor_to_shape


class ResBlock(EmbedBlock):
    """
    A residual block with optional upsampling or downsampling and skip connection.

    :param in_ch: Number of input channels.
    :param out_ch: Number of output channels.
    :param block_type: Type of block ("up", "down", or identity).
    :param kernel_size: Kernel size for convolution.
    :param padding: Padding size for convolution.
    :param embed_dim: Number of embedding channels (optional).
    :param dropout: Dropout rate (optional).
    :param use_conv: Whether to use a convolution in the skip connection.
    :param use_scale_shift_norm: Whether to use scale-shift normalization.
    """

    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 block_type: str,
                 kernel_size: Tuple[int, int, int],
                 padding: Tuple[int, int, int],
                 embed_dim: Optional[int] = None,
                 dropout: float = 0.0,
                 use_conv: bool = False,
                 use_scale_shift_norm: bool = False):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.emb_channels = embed_dim
        self.dropout = dropout
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm

        # Define the input transformation layers
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            nn.SiLU(),
            nn.Conv3d(in_ch, self.out_ch, kernel_size, padding=padding),
        )

        # Set up/down-sampling based on block type
        self.updown = block_type in {"up", "down"}
        if block_type == "up":
            self.h_upd = Upsample(in_ch, in_ch, kernel_size, padding)
            self.x_upd = Upsample(in_ch, in_ch, kernel_size, padding)
        elif block_type == "down":
            self.h_upd = Downsample(in_ch, in_ch, kernel_size, padding)
            self.x_upd = Downsample(in_ch, in_ch, kernel_size, padding)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        # Define embedding layers if embedding channels are provided
        if self.emb_channels:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(
                    self.emb_channels,
                    2 * self.out_ch if use_scale_shift_norm else self.out_ch,
                ),
            )

        # Define the output layers including dropout and convolution
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_ch),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv3d(self.out_ch, self.out_ch, kernel_size, padding=padding)
        )

        # Define the skip connection layer
        if self.out_ch == in_ch:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv3d(in_ch, self.out_ch, kernel_size, padding=padding)
        else:
            self.skip_connection = nn.Conv3d(in_ch, self.out_ch, 1)

    def forward(self, x: torch.Tensor, emb: Optional[Dict] = None, mask: Optional[torch.Tensor] = None,
                cond: Optional[torch.Tensor] = None, *args) -> torch.Tensor:
        """
        Forward pass for the ResBlock.

        :param x: Input tensor.
        :param emb: Optional embedding tensor.
        :param mask: Optional mask tensor.
        :param cond: Optional conditioning tensor.
        :return: Output tensor after applying the residual block with skip connection.
        """
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)  # Apply normalization and activation
            h = self.h_upd(h)  # Upsample or downsample if specified
            x = self.x_upd(x)  # Apply the same operation to the input
            h = in_conv(h)  # Apply final convolution
        else:
            h = self.in_layers(x)  # Apply input layers directly

        if self.emb_channels:
            emb_out = self.emb_layers(emb).type(h.dtype)
            emb_out = expand_tensor_to_shape(emb_out, h.shape, keep_dims=[0, 1])

            # Apply scale-shift normalization if configured
            if self.use_scale_shift_norm:
                out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
                scale, shift = torch.chunk(emb_out, 2, dim=1)
                h = out_norm(h) * (1 + scale) + shift
                h = out_rest(h)
            else:
                h = h + emb_out  # Direct addition without scale-shift

        if not self.use_scale_shift_norm:
            h = self.out_layers(h)

        # Combine output with skip connection
        skip = self.skip_connection(x)
        return skip + h


class ResBlockSequential(EmbedBlock):
    """
    A sequential container for multiple ResBlocks, allowing channel transformations and up/down-sampling.

    :param in_ch: Number of input channels.
    :param out_ch: List of output channels for each block.
    :param blocks: List of types of block ("up", "down", or identity).
    :param kernel_size: Kernel size for each block.
    :param embed_dim: Number of embedding channels.
    :param dropout: Dropout rate.
    :param use_conv: Whether to use convolution in the skip connection.
    :param use_scale_shift_norm: Whether to use scale-shift normalization.
    """

    def __init__(self, in_ch: int, out_ch: List[int], blocks: Union[str, List[str]],
                 kernel_size: Union[Tuple[int, int, int], List[Tuple[int, int, int]]] = (1, 3, 3),
                 embed_dim: Optional[Union[int, List[int]]] = None, dropout: Union[float, List[float]] = 0.0,
                 use_conv: Union[bool, List[bool]] = False, use_scale_shift_norm: Union[bool, List[bool]] = False):
        super().__init__()
        if isinstance(blocks, str):
            blocks = [blocks]  # Ensure blocks is a list
        if isinstance(kernel_size, tuple):
            kernel_size = len(blocks) * [kernel_size]  # Replicate kernel size for each block

        res_blocks = []
        for i, block in enumerate(blocks):
            padding = tuple(kernel_size[i][j] // 2 for j in range(len(kernel_size[i])))
            res_blocks.append(ResBlock(
                in_ch,
                out_ch if isinstance(out_ch, int) else out_ch[i],
                block,
                kernel_size[i],
                padding,
                embed_dim if isinstance(embed_dim, int) or embed_dim is None else embed_dim.pop(0),
                dropout if isinstance(dropout, float) else dropout.pop(0),
                use_conv if isinstance(use_conv, bool) else use_conv.pop(0),
                use_scale_shift_norm if isinstance(use_scale_shift_norm, bool) else use_scale_shift_norm.pop(0)
            ))
            in_ch = out_ch if isinstance(out_ch, int) else out_ch[i]
        # Sequential container for ResBlocks
        self.res_blocks = EmbedBlockSequential(*res_blocks)

    def forward(self, x: torch.Tensor, emb: Optional[Dict] = None, mask: Optional[torch.Tensor] = None,
                cond: Optional[torch.Tensor] = None, *args) -> torch.Tensor:
        """
        Forward pass for the ResBlockSequential.

        :param x: Input tensor.
        :param emb: Optional embedding tensor.
        :param mask: Optional mask tensor.
        :param cond: Optional conditioning tensor.
        :return: Output tensor after passing through the sequence of residual blocks.
        """
        return self.res_blocks(x, emb, mask, cond)
