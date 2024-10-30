import torch
from einops import rearrange

from stableclimgen.src.modules.cnn.resnet import EmbedBlock


class RearrangeBlock(EmbedBlock):
    def __init__(self, fn, pattern, reverse_pattern, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fn = fn
        self.pattern = pattern
        self.reverse_pattern = reverse_pattern

    def forward(self, x, emb=None, mask=None, cond=None, coords=None):
        b, v, t, g, c = x.shape

        for tensor in (x, emb, mask, cond, coords):
            if torch.is_tensor(tensor):
                tensor = rearrange(tensor, self.pattern)
        if isinstance(self.fn, EmbedBlock):
            x = self.fn(x, emb, mask, cond, coords)
        else:
            x = self.fn(x)
        x = rearrange(x, self.reverse_pattern, b=b, v=v, t=t, g=g, c=c)
        return x


class RearrangeTimeCentric(RearrangeBlock):
    def __init__(self, fn):
        super().__init__(fn, pattern='b v t g c -> (b v g) t c', reverse_pattern='(b v g) t c -> b v t g c')


class RearrangeSpaceCentric(RearrangeBlock):
    def __init__(self, fn):
        super().__init__(fn, pattern='b v t g c -> (b v t) g c', reverse_pattern='(b v t) g c -> b v t g c')


class RearrangeVarCentric(RearrangeBlock):
    def __init__(self, fn):
        super().__init__(fn, pattern='b v t g c -> (b t g) v c', reverse_pattern='(b t g) v c -> b v t g c')
