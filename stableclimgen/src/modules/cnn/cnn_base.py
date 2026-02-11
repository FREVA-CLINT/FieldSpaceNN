from abc import abstractmethod
from typing import Optional, Sequence, Tuple, Union, Dict, Any
from einops import rearrange
import torch.nn.functional as F

import torch
import torch.nn as nn

from ...utils.helpers import check_value
from ...modules.embedding.embedder import EmbedderSequential

def conv_nd(
    in_channels: int,
    out_channels: Optional[int],
    kernel_size: int | Tuple[int, int, int],
    stride: int | Tuple[int, int, int] = 1,
    padding: int | Tuple[int, int, int] = 0,
    dims: int = 3,
) -> nn.Module:
    """
    Create an N-D convolution module.

    :param in_channels: Number of input channels.
    :param out_channels: Number of output channels (defaults to ``in_channels``).
    :param kernel_size: Kernel size per spatial dimension.
    :param stride: Stride per spatial dimension.
    :param padding: Padding per spatial dimension.
    :param dims: Spatial dimensionality (1, 2, or 3).
    :return: Convolution module.
    """
    assert dims == 1 or dims == 2 or dims == 3
    return getattr(nn, f"Conv{dims}d")(in_channels, out_channels or in_channels, kernel_size, stride, padding)


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
        self.use_conv: bool = use_conv
        # Define upsampling operation, using nearest-neighbor if none is provided
        self.upsample: nn.Module = upsample or nn.Upsample(scale_factor=tuple(x + 1 for x in padding), mode="nearest")
        # Define convolution layer if use_conv is True
        if use_conv:
            self.conv: nn.Module = conv_nd(in_channels, out_channels, kernel_size, 1, padding, dims)

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
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels or in_channels
        self.use_conv: bool = use_conv
        stride = 2  # Define stride for downsampling in height and width dimensions
        # Select convolution or average pooling for downsampling
        if use_conv:
            self.op: nn.Module = conv_nd(self.in_channels, self.out_channels, kernel_size, stride, padding, dims)
        else:
            self.op: nn.Module = nn.AvgPool3d(kernel_size=stride, stride=stride)

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

    def __init__(
        self,
        in_ch: int,
        out_ch: Sequence[int],
        blocks: Union[str, Sequence[str]],
        kernel_size: int | Sequence[int] | Sequence[Sequence[int]] = 3,
        dims: int = 2,
        **kwargs,
    ) -> None:
        """
        Initialize the convolutional block sequence.

        :param in_ch: Number of input channels.
        :param out_ch: Output channels per block.
        :param blocks: Block types ("up", "down", or identity).
        :param kernel_size: Kernel sizes per block.
        :param dims: Spatial dimensionality (1, 2, or 3).
        :param kwargs: Additional arguments (unused).
        :return: None.
        """
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
        self.conv_blocks: nn.Module = EmbedBlockSequential(*conv_blocks)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass through the sequence of convolutional blocks.

        :param x: Input tensor of shape ``(b, c, h, w)`` or ``(b, c, d, h, w)``.
        :return: Output tensor after applying the sequence of blocks.
        """
        return self.conv_blocks(x)  # Apply each block in sequence to the input tensor


class EmbedBlock(nn.Module):
    """
    Abstract base module where `forward()` takes timestep embeddings as a second argument.
    This class provides a standard interface for modules that operate with embeddings.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor, emb: Optional[Dict], mask: Optional[torch.Tensor],
                *args, **kwargs) -> torch.Tensor:
        """
        Apply the module to `x` given `emb`, `mask`, `cond`, `coords`.

        :param x: Input tensor.
        :param emb: Embedding tensor, providing different embedding information.
        :param mask: Optional mask tensor.
        :param cond: Optional conditioning tensor.
        :return: Processed tensor after applying the module.
        """
        pass


class EmbedBlockSequential(nn.Sequential, EmbedBlock):
    """
    A sequential module that passes timestep embeddings to child modules that
    support them as an extra input. This class extends `nn.Sequential` to handle
    embedding-based layers within a sequence.
    """

    def forward(self, x: torch.Tensor, emb: Optional[Dict] = None,
                mask: Optional[torch.Tensor] = None, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass for the EmbedBlockSequential.

        :param x: Input tensor.
        :param emb: Embedding tensor, providing different embedding information.
        :param mask: Optional mask tensor to be passed to layers that support it.
        :param cond: Optional conditioning tensor to be passed to layers that support it.
        :param coords: Optional coordinates tensor to be passed to layers that support it.
        :return: Output tensor after sequentially applying each layer, with embeddings if supported.
        """
        for layer in self:
            if isinstance(layer, EmbedBlock):
                x = layer(x, emb, mask, *args)
            else:
                x = layer(x)
        return x
    

class PatchEmbedderND(EmbedBlock):
    """
    Embeds a 3D patch from the input using a convolutional layer.

    :param in_channels: Number of input channels.
    :param out_channels: Number of output channels.
    :param kernel_size: Size of the convolutional kernel as a tuple of three integers.
    :param patch_size: Size of the patch as a tuple of three integers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int],
        patch_size: Tuple[int, int, int],
        dims: int = 2
    ) -> None:
        """
        Initialize the patch embedder.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param kernel_size: Size of the convolutional kernel.
        :param patch_size: Size of the patch.
        :param dims: Number of spatial dimensions.
        :return: None.
        """
        super().__init__()
        assert len(patch_size) == len(kernel_size) == dims
        # Compute padding to match patch size
        padding = tuple((kernel_size[i] - patch_size[i]) // 2 for i in range(len(kernel_size)))
        self.conv: nn.Module = conv_nd(in_channels, out_channels, kernel_size, patch_size, padding, dims=dims)
        self.dims: int = dims

    def forward(
        self,
        x: torch.Tensor,
        emb: Optional[Dict[str, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        *args: Any
    ) -> torch.Tensor:
        """
        Forward pass through the patch embedding layer.

        :param x: Input tensor of shape ``(b, v, t, n, d, f)`` or a compatible subset.
        :param emb: Optional embedding tensor.
        :param mask: Optional mask tensor.
        :param cond: Optional conditioning tensor.
        :return: Embedded output tensor with patches along spatial dimensions.
        """
        x = self.conv(x)
        return x


class ConvUnpatchify(EmbedBlock):
    """
    Converts patches back to the original shape using a convolutional layer.

    :param in_channels: Number of input channels.
    :param out_channels: Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int, dims: int = 2, kernel_size: int = 3) -> None:
        """
        Initialize the convolutional unpatchifier.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param dims: Number of spatial dimensions.
        :param kernel_size: Convolution kernel size.
        :return: None.
        """
        super().__init__()
        # Convolution to reconstruct output shape
        self.dims: int = dims
        kernel_size = check_value(kernel_size, dims)
        padding = tuple(k // 2 for k in kernel_size)
        self.out: nn.Module = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            conv_nd(in_channels, out_channels, kernel_size, 1, padding, dims=dims)
        )

    def forward(
        self,
        x: torch.Tensor,
        emb: Optional[Dict[str, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        out_shape: Optional[Tuple[int, int, int]] = None,
        *args: Any
    ) -> torch.Tensor:
        """
        Forward pass through the unpatchifying layer with optional interpolation.

        :param x: Input tensor of shape ``(b, v, t, n, d, f)`` or a compatible subset.
        :param emb: Optional embedding tensor.
        :param mask: Optional mask tensor.
        :param cond: Optional conditioning tensor.
        :param out_shape: Target output shape for interpolation.
        :return: Reconstructed output tensor with the requested spatial shape.
        """
        x = F.interpolate(x, size=out_shape, mode="nearest")  # Resize to target shape
        return self.out(x)


class LinearUnpatchify(EmbedBlock):
    """
    Converts linear embeddings back to the original shape.

    :param in_channels: Number of input channels.
    :param out_channels: Number of output channels.
    :param patch_size: Size of the patches as a tuple of three integers.
    :param embed_dim: Embedding dimension, if applicable.
    :param spatial_dim_count: Number of spatial dimensions (2 or 3).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: Tuple[int, int, int],
        embedder: Optional[EmbedderSequential] = None,
        embed_confs: Optional[Dict[str, Any]] = None,
        embed_mode: str = "sum",
        spatial_dim_count: int = 1
    ) -> None:
        """
        Initialize the linear unpatchifier.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param patch_size: Patch size along each spatial dimension.
        :param embedder: Optional embedder for scale-shift conditioning.
        :param embed_confs: Optional embedder configuration.
        :param embed_mode: Embedding combination mode.
        :param spatial_dim_count: Number of spatial dimensions represented by "s".
        :return: None.
        """
        super().__init__()
        # Normalization layer for linear unpatchifying
        self.norm: nn.Module = nn.LayerNorm(in_channels, elementwise_affine=False, eps=1e-6)
        # Linear layer to reconstruct spatial dimensions
        self.linear: nn.Module = nn.Linear(
            in_channels,
            patch_size[0] * patch_size[1] * patch_size[2] * out_channels,
            bias=True,
        )
        self.patch_size: Tuple[int, int, int] = patch_size
        self.out_channels: int = out_channels
        self.spatial_dim_count: int = spatial_dim_count
        self.embedder_seq: Optional[EmbedderSequential] = embedder
        self.embedding_layer: Optional[nn.Module] = None
        # Optional embedding layer
        if embedder:
            self.embedding_layer = torch.nn.Linear(self.embedder_seq.get_out_channels, 2 * in_channels)


    def forward(
        self,
        x: torch.Tensor,
        emb: Optional[Dict[str, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        out_shape: Optional[Tuple[int, int, int]] = None,
        *args: Any
    ) -> torch.Tensor:
        """
        Forward pass to reconstruct the linear embeddings back to original dimensions.

        :param x: Input tensor of shape ``(b, ..., v, c)``.
        :param emb: Optional embedding tensor for scale-shift normalization.
        :param mask: Optional mask tensor.
        :param cond: Optional conditioning tensor.
        :param out_shape: Target output shape for the unpatchifying operation.
        :return: Reconstructed output tensor with shape ``(b, ..., v, c)``.
        """
        b, v = x.shape[0], x.shape[-2]
        if hasattr(self, 'embedding_layer'):
            scale, shift = self.embedding_layer(self.embedder_seq(emb)).chunk(2, dim=-1)
            x = self.norm(x) * (scale + 1) + shift
        else:
            x = self.norm(x)
        x = rearrange(x, 'b ... v c -> (b v) (...) c')  # Flatten dimensions for linear layer
        x = self.linear(x)
        return rearrange(self.unpatchify(x, out_shape), '(b v) ... c -> b ... v c', b=b, v=v)

    def unpatchify(self, x: torch.Tensor, out_shape: Tuple[int, int, int]) -> torch.Tensor:
        """
        Reshape the output tensor to the original dimensions after linear unpatchifying.

        :param x: Input tensor to be reshaped.
        :param out_shape: Target output shape as a tuple of three integers.
        :return: Tensor reshaped to the original dimensions.
        """
        c = self.out_channels
        p = self.patch_size
        if self.spatial_dim_count == 2:
            t, h, w = tuple(o // p_i for o, p_i in zip(out_shape, p))
            x = x.reshape(shape=(x.shape[0], t, h, w, p[0], p[1], p[2], c))
            x = torch.einsum('nthwpqrc->nctphqwr', x)  # Rearrange to match target shape
            return x.reshape(shape=(x.shape[0], t * p[0], h * p[1], w * p[2], c))
        else:
            t, g = tuple(o // p_i for o, p_i in zip(out_shape, p))
            x = x.reshape(shape=(x.shape[0], t, g, p[0], p[1], c))
            x = torch.einsum('ntgpqc->nctpgq', x)  # Rearrange to match target shape
            return x.reshape(shape=(x.shape[0], t * p[0], g * p[1], c))
