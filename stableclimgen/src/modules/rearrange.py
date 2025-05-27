from typing import Optional, Dict

from einops import rearrange
import torch

from stableclimgen.src.utils.utils import EmbedBlock
from stableclimgen.src.modules.base import IdentityLayer

class RearrangeBlock(EmbedBlock):
    """
    A block that rearranges input tensor shapes before and after applying a specified function.

    :param fn: The function to apply to the rearranged input.
    :param pattern: The rearrangement pattern for the input tensor.
    :param reverse_pattern: The reverse pattern to apply after the function is executed.
    :param spatial_dim_count: Determines the number of spatial dimensions, adjusting the rearrangement accordingly.
    """

    def __init__(self, fn, 
                 pattern: str, 
                 reverse_pattern: str, 
                 spatial_dim_count: int = 1, 
                 seq_len: int = None, 
                 proj_layer: torch.nn.Module = None,
                 out_layer: torch.nn.Module = None):
        
        super().__init__()
        self.fn = fn
        self.pattern = pattern
        self.reverse_pattern = reverse_pattern
        self.seq_len = seq_len
        self.proj_layer = proj_layer if proj_layer else IdentityLayer()
        self.out_layer = out_layer if out_layer else IdentityLayer()

        if spatial_dim_count == 2:
            # Replace the variable dimension 'g' with height and width dimensions for 2D spatial arrangements
            self.pattern = self.pattern.replace('g', 'h w')
            self.reverse_pattern = self.reverse_pattern.replace('g', 'h w')

    def forward(self, x: torch.Tensor, emb: Optional[Dict] = None, mask: Optional[torch.Tensor] = None,
                cond: Optional[torch.Tensor] = None, *args) -> torch.Tensor:
        """
        Forward pass for the RearrangeBlock.

        :param x: Input tensor with shape (batch, vars, time, height, width, channels) or (batch, vars, time, group, channels).
        :param emb: Optional embedding Dictionary.
        :param mask: Optional mask tensor.
        :param cond: Optional conditioning tensor.
        :return: Tensor after rearrangement, function application, and reverse rearrangement.
        """
        # Determine the input dimensionality and extract batch, time, and variable dimensions

        x = self.proj_layer(x, emb=emb)

        dim = x.dim()
        if dim == 6:
            b, t, h, w, v, c = x.shape
        else:
            b, t, g, v, c = x.shape


        # Rearrange input and optional tensors according to the specified pattern
        x, mask, cond = [
            rearrange(tensor, self.pattern) if torch.is_tensor(tensor) else tensor
            for tensor in (x, mask, cond)
        ]

        s = x.shape[1]
        if self.seq_len and self.seq_len < s:
            n = s // self.seq_len
            x, mask, cond = [
                rearrange(tensor, 'b (n s) ... -> (b n) s ...', n=n, s=self.seq_len) if torch.is_tensor(tensor) else tensor
                for tensor in (x, mask, cond)
            ]

        # Apply the specified function (fn) to the rearranged tensor
        if isinstance(self.fn, EmbedBlock):
            x = self.fn(x, emb, mask, cond, *args)
        else:
            x = self.fn(x, *args)

        if self.seq_len and self.seq_len < s:
            x, mask, cond = [
                rearrange(tensor, '(b n) s  ... -> b (n s)  ...', n=n, s=self.seq_len) if torch.is_tensor(tensor) else tensor
                for tensor in (x, mask, cond)
            ]

        # Reverse rearrange to restore the original shape
        if dim == 6:
            x = rearrange(x, self.reverse_pattern, b=b, t=t, h=h, w=w, v=v)
        else:
            x = rearrange(x, self.reverse_pattern, b=b, t=t, g=g, v=v)

        x = self.out_layer(x, emb=emb)
        return x


class RearrangeTimeCentric(RearrangeBlock):
    """
    Rearranges input to make the time dimension centric before applying a function, then reverses it.

    :param fn: The function to apply to the rearranged input.
    :param spatial_dim_count: Determines the number of spatial dimensions, adjusting the rearrangement accordingly.
    """

    def __init__(self, fn, spatial_dim_count=1, seq_len: int = None, proj_layer: torch.nn.Module = None, out_layer: torch.nn.Module = None):
        super().__init__(
            fn,
            pattern='b t g v c -> (b g v) t c',
            reverse_pattern='(b g v) t c -> b t g v c',
            spatial_dim_count=spatial_dim_count,
            seq_len=seq_len,
            proj_layer=proj_layer,
            out_layer=out_layer
        )


class RearrangeSpaceCentric(RearrangeBlock):
    """
    Rearranges input to make spatial dimensions centric before applying a function, then reverses it.

    :param fn: The function to apply to the rearranged input.
    :param spatial_dim_count: Determines the number of spatial dimensions, adjusting the rearrangement accordingly.
    """

    def __init__(self, fn, spatial_dim_count=1, seq_len: int = None, proj_layer: torch.nn.Module = None, out_layer: torch.nn.Module = None):
        super().__init__(
            fn,
            pattern='b t g v c -> (b t v) (g) c',
            reverse_pattern='(b t v) (g) c -> b t g v c',
            spatial_dim_count=spatial_dim_count,
            seq_len=seq_len,
            proj_layer=proj_layer,
            out_layer=out_layer
        )


class RearrangeVarCentric(RearrangeBlock):
    """
    Rearranges input to make the variable dimension centric before applying a function, then reverses it.

    :param fn: The function to apply to the rearranged input.
    :param spatial_dim_count: Determines the number of spatial dimensions, adjusting the rearrangement accordingly.
    """

    def __init__(self, fn, spatial_dim_count=1, seq_len: int = None, proj_layer: torch.nn.Module = None, out_layer: torch.nn.Module = None):
        super().__init__(
            fn,
            pattern='b t g v c -> (b t g) v c',
            reverse_pattern='(b t g) v c -> b t g v c',
            spatial_dim_count=spatial_dim_count,
            seq_len=seq_len,
            proj_layer=proj_layer,
            out_layer=out_layer
        )


class RearrangeConvCentric(RearrangeBlock):
    """
    Rearranges input to make the convolutional channels centric before applying a function, then reverses it.

    :param fn: The function to apply to the rearranged input.
    :param spatial_dim_count: Determines the number of spatial dimensions, adjusting the rearrangement accordingly.
    """

    def __init__(self, fn, spatial_dim_count=1, seq_len: int = None, dims: int = 2):
        assert dims == 1 or dims == 2 or dims == 3
        if dims - spatial_dim_count == 0:
            pattern = 'b t g v c -> (b t v) c g'
            reverse_pattern = '(b t v) c g -> b t g v c'
        elif dims - spatial_dim_count == 1:
            pattern = 'b t g v c -> (b v) c t g'
            reverse_pattern = '(b v) c t g -> b t g v c'
        else:
            pattern = 'b t g v c -> b c t g v'
            reverse_pattern = 'b c t g v -> b t g v c'

        super().__init__(
            fn,
            pattern=pattern,
            reverse_pattern=reverse_pattern,
            spatial_dim_count=spatial_dim_count,
            seq_len=seq_len
        )

    def forward(self, x: torch.Tensor, emb: Optional[Dict] = None, mask: Optional[torch.Tensor] = None,
                cond: Optional[torch.Tensor] = None, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass for the RearrangeConvCentric.

        :param x: Input tensor with shape (batch, time, height, width, vars, channels).
        :param emb: Optional embedding dict.
        :param mask: Optional mask tensor.
        :param cond: Optional conditioning tensor.
        :return: Tensor after rearrangement, function application, and reverse rearrangement.
        """
        b, t, v, c = x.shape[0], x.shape[1], x.shape[-2], x.shape[-1]

        # Rearrange input and optional tensors according to the specified pattern
        x, mask, cond = [
            rearrange(tensor, self.pattern) if torch.is_tensor(tensor) else tensor
            for tensor in (x, mask, cond)
        ]

        # Apply the function (fn) to the rearranged tensor
        if isinstance(self.fn, EmbedBlock):
            x = self.fn(x, emb, mask, cond, *args, **kwargs)
        else:
            x = self.fn(x, *args, **kwargs)

        # Reverse rearrange to restore the original shape
        return rearrange(x, self.reverse_pattern, b=b, v=v)
