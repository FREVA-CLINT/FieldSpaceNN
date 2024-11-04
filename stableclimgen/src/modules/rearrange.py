from typing import Optional

from einops import rearrange
import torch

from stableclimgen.src.modules.utils import EmbedBlock


class RearrangeBlock(EmbedBlock):
    """
    A block that rearranges input tensor shapes before and after applying a specified function.

    :param fn: The function to apply to the rearranged input.
    :param pattern: The rearrangement pattern for the input tensor.
    :param spatial_dim_count: Determines the number of the spatial dimensions and adjusts rearrange
    :param reverse_pattern: The reverse pattern to apply after the function is executed.
    """

    def __init__(self, fn, pattern: str, reverse_pattern: str, spatial_dim_count: int = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fn = fn
        self.pattern = pattern
        self.reverse_pattern = reverse_pattern
        if spatial_dim_count == 2:
            self.pattern = self.pattern.replace('g', 'h w')
            self.reverse_pattern = self.reverse_pattern.replace('g', 'h w')


    def forward(self, x: torch.Tensor, emb: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None,
                cond: Optional[torch.Tensor] = None, coords: Optional[torch.Tensor] = None, *args) -> torch.Tensor:
        """
        Forward pass for the RearrangeBlock.

        :param x: Input tensor with shape (batch, vars, time, height, width, channels).
        :param emb: Optional embedding tensor.
        :param mask: Optional mask tensor.
        :param cond: Optional conditioning tensor.
        :param coords: Optional coordinates tensor.
        :return: Tensor after rearrangement, function application, and reverse rearrangement.
        """
        dim = x.dim()
        if dim == 6:
            b, t, h, w, v, c = x.shape
        else:
            b, t, g, v, c = x.shape

        x, emb, mask, cond = [
            rearrange(tensor, self.pattern) if torch.is_tensor(tensor) else tensor
            for tensor in (x, emb, mask, cond)
        ]

        # Apply the function (fn) to the rearranged x
        if isinstance(self.fn, EmbedBlock):
            x = self.fn(x, emb, mask, cond, coords, *args)
        else:
            x = self.fn(x, *args)

        if dim == 6:
            x = rearrange(x, self.reverse_pattern, b=b, t=t, h=h, w=w, v=v) # need to fix
        else:
            x = rearrange(x, self.reverse_pattern, b=b, t=t, g=g, v=v)
        return x


class RearrangeTimeCentric(RearrangeBlock):
    """
    Rearranges input to make time dimension centric before applying a function, then reverses.

    :param fn: The function to apply to the rearranged input.
    :param spatial_dim_count: Determines the number of the spatial dimensions and adjusts rearrange
    """

    def __init__(self, fn, spatial_dim_count=1):
        super().__init__(fn, pattern='b t g  v c -> (b g v) t c', reverse_pattern='(b g v) t c -> b t g v c', spatial_dim_count=spatial_dim_count)


class RearrangeSpaceCentric(RearrangeBlock):
    """
    Rearranges input to make spatial dimensions centric before applying a function, then reverses.

    :param fn: The function to apply to the rearranged input.
    :param spatial_dim_count: Determines the number of the spatial dimensions and adjusts rearrange
    """

    def __init__(self, fn, spatial_dim_count=1):
        super().__init__(fn, pattern='b t g v c -> (b t v) (g) c', reverse_pattern='(b t v) (g) c -> b t g v c', spatial_dim_count=spatial_dim_count)


class RearrangeVarCentric(RearrangeBlock):
    """
    Rearranges input to make variable dimension centric before applying a function, then reverses.

    :param fn: The function to apply to the rearranged input.
    :param spatial_dim_count: Determines the number of the spatial dimensions and adjusts rearrange
    """

    def __init__(self, fn, spatial_dim_count=1):
        super().__init__(fn, pattern='b t g v c -> (b t g) v c', reverse_pattern='(b t g) v c -> b t g v c', spatial_dim_count=spatial_dim_count)


class RearrangeConvCentric(RearrangeBlock):
    """
    Rearranges input to make convolution dimension centric before applying a function, then reverses.

    :param fn: The function to apply to the rearranged input.
    :param spatial_dim_count: Determines the number of the spatial dimensions and adjusts rearrange
    """

    def __init__(self, fn, spatial_dim_count=1):
        super().__init__(fn, pattern='b t g v c -> (b v) c t g', reverse_pattern='(b v) c t g -> b t g v c', spatial_dim_count=spatial_dim_count)

    def forward(self, x: torch.Tensor, emb: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None,
                cond: Optional[torch.Tensor] = None, coords: Optional[torch.Tensor] = None, *args) -> torch.Tensor:
        """
        Forward pass for the RearrangeConvCentric.

        :param x: Input tensor with shape (batch, vars, time, height, width, channels).
        :param emb: Optional embedding tensor.
        :param mask: Optional mask tensor.
        :param cond: Optional conditioning tensor.
        :param coords: Optional coordinates tensor.
        :return: Tensor after rearrangement, function application, and reverse rearrangement.
        """
        b, t, h, w, v, c = x.shape

        x, emb, mask, cond = [
            rearrange(tensor, self.pattern) if torch.is_tensor(tensor) else tensor
            for tensor in (x, emb, mask, cond)
        ]
        # Apply the function (fn) to the rearranged x
        if isinstance(self.fn, EmbedBlock):
            x = self.fn(x, emb, mask, cond, coords, *args)
        else:
            x = self.fn(x, *args)

        return rearrange(x, self.reverse_pattern, b=b, t=t, v=v)
