from typing import Optional

from einops import rearrange
import torch

from stableclimgen.src.modules.utils import EmbedBlock


class RearrangeBlock(EmbedBlock):
    """
    A block that rearranges input tensor shapes before and after applying a specified function.

    :param fn: The function to apply to the rearranged input.
    :param pattern: The rearrangement pattern for the input tensor.
    :param reverse_pattern: The reverse pattern to apply after the function is executed.
    """

    def __init__(self, fn, pattern: str, reverse_pattern: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fn = fn
        self.pattern = pattern
        self.reverse_pattern = reverse_pattern

    def forward(self, x: torch.Tensor, emb: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None,
                cond: Optional[torch.Tensor] = None, coords: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        Forward pass for the RearrangeBlock.

        :param x: Input tensor with shape (batch, vars, time, height, width, channels).
        :param emb: Optional embedding tensor.
        :param mask: Optional mask tensor.
        :param cond: Optional conditioning tensor.
        :param coords: Optional coordinates tensor.
        :return: Tensor after rearrangement, function application, and reverse rearrangement.
        """
        b, v, t, h, w, c = x.shape

        x, emb, mask, cond = [
            rearrange(tensor, self.pattern) if torch.is_tensor(tensor) else tensor
            for tensor in (x, emb, mask, cond)
        ]

        # Apply the function (fn) to the rearranged x
        if isinstance(self.fn, EmbedBlock):
            x = self.fn(x, emb, mask, cond, coords, **kwargs)
        else:
            x = self.fn(x, **kwargs)

        return rearrange(x, self.reverse_pattern, b=b, v=v, t=t, h=h, w=w)


class RearrangeTimeCentric(RearrangeBlock):
    """
    Rearranges input to make time dimension centric before applying a function, then reverses.

    :param fn: The function to apply to the rearranged input.
    """

    def __init__(self, fn):
        super().__init__(fn, pattern='b v t h w c -> (b v h w) t c', reverse_pattern='(b v h w) t c -> b v t h w c')


class RearrangeSpaceCentric(RearrangeBlock):
    """
    Rearranges input to make spatial dimensions centric before applying a function, then reverses.

    :param fn: The function to apply to the rearranged input.
    """

    def __init__(self, fn):
        super().__init__(fn, pattern='b v t h w c -> (b v t) (h w) c', reverse_pattern='(b v t) (h w) c -> b v t h w c')


class RearrangeVarCentric(RearrangeBlock):
    """
    Rearranges input to make variable dimension centric before applying a function, then reverses.

    :param fn: The function to apply to the rearranged input.
    """

    def __init__(self, fn):
        super().__init__(fn, pattern='b v ... c -> (b ...) v c', reverse_pattern='(b ...) v c -> b v ... c')


class RearrangeConvCentric(RearrangeBlock):
    """
    Rearranges input to make convolution dimension centric before applying a function, then reverses.

    :param fn: The function to apply to the rearranged input.
    """

    def __init__(self, fn):
        super().__init__(fn, pattern='b v t h w c -> (b v) c t h w', reverse_pattern='(b v) c t h w -> b v t h w c')

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
        b, v, t, h, w, c = x.shape
        x, emb, mask, cond = [
            rearrange(tensor, self.pattern) if torch.is_tensor(tensor) else tensor
            for tensor in (x, emb, mask, cond)
        ]

        # Apply the function (fn) to the rearranged x
        if isinstance(self.fn, EmbedBlock):
            x = self.fn(x, emb, mask, cond, coords, *args)
        else:
            x = self.fn(x, **kwargs)

        return rearrange(x, self.reverse_pattern, b=b, v=v, t=t, h=h, w=w)
