from typing import Optional, Sequence, Tuple, Union, Dict

import torch
import torch.nn as nn

from einops import rearrange

from fieldspacenn.src.modules.cnn.cnn_base import Upsample, Downsample, conv_nd, EmbedBlock, EmbedBlockSequential
from fieldspacenn.src.utils.helpers import check_value


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
                 kernel_size: int | Tuple,
                 padding: int | Tuple,
                 embedder: Optional[EmbedBlockSequential],
                 dropout: float = 0.0,
                 use_conv: bool = False,
                 use_scale_shift_norm: bool = False,
                 dims: int = 2):
        super().__init__()
        self.dropout: float = dropout
        self.use_conv: bool = use_conv
        self.use_scale_shift_norm: bool = use_scale_shift_norm
        self.dims: int = dims

        # Define the input transformation layers
        self.in_layers: nn.Sequential = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            nn.SiLU(),
            conv_nd(in_ch, out_ch, kernel_size, padding=padding, dims=dims),
        )

        # Set up/down-sampling based on block type
        self.updown: bool = block_type in {"up", "down"}
        if block_type == "up":
            self.h_upd: nn.Module = Upsample(in_ch, in_ch, kernel_size, padding, dims=dims)
            self.x_upd: nn.Module = Upsample(in_ch, in_ch, kernel_size, padding, dims=dims)
        elif block_type == "down":
            self.h_upd = Downsample(in_ch, in_ch, kernel_size, padding, dims=dims)
            self.x_upd = Downsample(in_ch, in_ch, kernel_size, padding, dims=dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()


        if embedder is not None:
            self.embedder_seq: EmbedBlockSequential = embedder
            self.embedding_layer: nn.Linear = torch.nn.Linear(
                self.embedder_seq.get_out_channels,
                out_ch * (2 ** use_scale_shift_norm),
            )

        # Define the output layers including dropout and convolution
        self.out_layers: nn.Sequential = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            conv_nd(out_ch, out_ch, kernel_size, padding=padding, dims=dims)
        )

        # Define the skip connection layer
        if out_ch == in_ch:
            self.skip_connection: nn.Module = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(in_ch, out_ch, kernel_size, padding=padding, dims=dims)
        else:
            self.skip_connection = conv_nd(in_ch, out_ch, 1, dims=dims)

    def forward(
        self,
        x: torch.Tensor,
        emb: Optional[Dict] = None,
        mask: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        *args,
    ) -> torch.Tensor:
        """
        Forward pass for the ResBlock.

        :param x: Input tensor of shape ``(b, c, h, w)`` or ``(b, c, d, h, w)``.
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

        if hasattr(self, "embedding_layer"):
            emb_out = self.embedding_layer(self.embedder_seq(emb))
            pattern = 'b t h w v c -> (b v) c t h w' if self.dims == 3 else 'b v t h w c -> (b v t) c h w'
            emb_out = rearrange(emb_out, pattern)

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

    def __init__(
        self,
        in_ch: int,
        out_ch: Sequence[int],
        blocks: Union[str, Sequence[str]],
        kernel_size: int | Sequence[int] | Sequence[Sequence[int]] = 3,
        embedders: Optional[Sequence[EmbedBlockSequential]] = None,
        dropout: Union[float, Sequence[float]] = 0.0,
        use_conv: Union[bool, Sequence[bool]] = False,
        use_scale_shift_norm: Union[bool, Sequence[bool]] = False,
        dims: int = 2,
        **kwargs,
    ) -> None:
        """
        Initialize the residual block sequence.

        :param in_ch: Number of input channels.
        :param out_ch: Output channels per block.
        :param blocks: Block types ("up", "down", or identity).
        :param kernel_size: Kernel sizes per block.
        :param embedders: Optional embedders per block.
        :param dropout: Dropout rates per block.
        :param use_conv: Whether to use convolution in skip connections.
        :param use_scale_shift_norm: Whether to use scale-shift normalization.
        :param dims: Spatial dimensionality (1, 2, or 3).
        :param kwargs: Additional arguments (unused).
        :return: None.
        """
        super().__init__()
        out_ch = check_value(out_ch, len(blocks))
        kernel_size = check_value(kernel_size, len(blocks))
        dropout = check_value(dropout, len(blocks))
        use_conv = check_value(use_conv, len(blocks))
        use_scale_shift_norm = check_value(use_scale_shift_norm, len(blocks))
        res_blocks = []
        for i, block in enumerate(blocks):
            kernel_size[i] = check_value(kernel_size[i], dims)
            padding = [kernel_size[i][j] // 2 for j in range(len(kernel_size[i]))]
            res_blocks.append(ResBlock(
                in_ch,
                out_ch[i],
                block,
                kernel_size[i],
                padding,
                embedders[i] if embedders else None,
                dropout[i],
                use_conv[i],
                use_scale_shift_norm[i],
                dims
            ))
            in_ch = out_ch if isinstance(out_ch, int) else out_ch[i]
        # Sequential container for ResBlocks
        self.res_blocks: nn.Module = EmbedBlockSequential(*res_blocks)

    def forward(
        self,
        x: torch.Tensor,
        emb: Optional[Dict] = None,
        mask: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass for the ResBlockSequential.

        :param x: Input tensor of shape ``(b, c, h, w)`` or ``(b, c, d, h, w)``.
        :param emb: Optional embedding tensor.
        :param mask: Optional mask tensor.
        :param cond: Optional conditioning tensor.
        :return: Output tensor after passing through the sequence of residual blocks.
        """
        return self.res_blocks(x, emb, mask, cond)
