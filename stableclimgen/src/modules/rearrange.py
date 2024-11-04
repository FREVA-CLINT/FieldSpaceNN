from einops import rearrange
import torch

from stableclimgen.src.modules.utils import EmbedBlock


class RearrangeBlock(EmbedBlock):
    def __init__(self, fn, pattern, reverse_pattern, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fn = fn
        self.pattern = pattern
        self.reverse_pattern = reverse_pattern

    def forward(self, x, emb=None, mask=None, cond=None, coords=None, **kwargs):
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
    def __init__(self, fn):
        super().__init__(fn, pattern='b v t h w c -> (b v h w) t c', reverse_pattern='(b v h w) t c -> b v t h w c')


class RearrangeSpaceCentric(RearrangeBlock):
    def __init__(self, fn):
        super().__init__(fn, pattern='b v t h w c -> (b v t) (h w) c', reverse_pattern='(b v t) (h w) c -> b v t h w c')


class RearrangeVarCentric(RearrangeBlock):
    def __init__(self, fn):
        super().__init__(fn, pattern='b v ... c -> (b ...) v c', reverse_pattern='(b ...) v c -> b v ... c')


class RearrangeConvCentric(RearrangeBlock):
    def __init__(self, fn):
        super().__init__(fn, pattern='b v t h w c -> (b v) c t h w', reverse_pattern='(b v) c t h w -> b v t h w c')

    def forward(self, x, emb=None, mask=None, cond=None, coords=None, *args):
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

        return rearrange(x, self.reverse_pattern, b=b, v=v, t=t)