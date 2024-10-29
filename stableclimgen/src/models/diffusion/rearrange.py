import torch
from einops import rearrange

from stableclimgen.src.modules.cnn.resnet import EmbedBlock


class RearrangeTimeCentric(EmbedBlock):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, emb=None, mask=None, cond=None, **kwargs):
        b, c, t, w, h = x.shape
        x = rearrange(x, 'b c t w h -> (b w h) t c')
        if torch.is_tensor(mask):
            mask = rearrange(mask, 'b c t w h -> (b w h) t c')
        if torch.is_tensor(cond):
            cond = rearrange(cond, 'b c t w h -> (b w h) t c')
        if torch.is_tensor(emb):
            emb = rearrange(emb, 'b c t w h -> (b w h) t c')
        if isinstance(self.fn, EmbedBlock):
            x = self.fn(x, emb, mask, cond, **kwargs)
        else:
            x = self.fn(x, **kwargs)
        x = rearrange(x, '(b w h) t c -> b c t w h', b=b, c=c, t=t, w=w, h=h)
        return x


class RearrangeSpaceCentric(EmbedBlock):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, emb=None, mask=None, cond=None, **kwargs):
        b, c, t, w, h = x.shape
        x = rearrange(x, 'b c t w h -> (b t) (w h) c')
        if torch.is_tensor(mask):
            mask = rearrange(mask, 'b c t w h -> (b t) (w h) c')
        if torch.is_tensor(cond):
            cond = rearrange(cond, 'b c t w h -> (b t) (w h) c')
        if torch.is_tensor(emb):
            emb = rearrange(emb, 'b c t w h -> (b t) (w h) c')
        if isinstance(self.fn, EmbedBlock):
            x = self.fn(x, emb, mask, cond, **kwargs)
        else:
            x = self.fn(x, **kwargs)
        x = rearrange(x, '(b t) (w h) c -> b c t w h', b=b, c=c, t=t, w=w, h=h)
        return x


class RearrangeBatchCentric(EmbedBlock):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, emb=None, mask=None, cond=None, **kwargs):
        t = x.shape[2]
        x = rearrange(x, 'b c t ... -> (b t) c ...')
        if torch.is_tensor(mask):
            mask = rearrange(mask, 'b c t ... -> (b t) c ...')
        if torch.is_tensor(cond):
            cond = rearrange(cond, 'b c t ... -> (b t) c ...')
        if torch.is_tensor(emb):
            emb = rearrange(emb, 'b c t ... -> (b t) c ...')
        if isinstance(self.fn, EmbedBlock):
            x = self.fn(x, emb, mask, cond, **kwargs)
        else:
            x = self.fn(x, **kwargs)
        x = rearrange(x, '(b t) c ... -> b c t ...', t=t)
        return x


class RearrangeSpaceChannelCentric(EmbedBlock):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, emb=None, mask=None, cond=None, **kwargs):
        b, c, t, w, h = x.shape
        x = rearrange(x, 'b c t w h -> (b t) c (w h)')
        if torch.is_tensor(mask):
            mask = rearrange(mask, 'b c t w h -> (b t) c (w h)')
        if torch.is_tensor(cond):
            cond = rearrange(cond, 'b c t w h -> (b t) c (w h)')
        if torch.is_tensor(emb):
            emb = rearrange(emb, 'b c t w h -> (b t) c (w h)')
        if isinstance(self.fn, EmbedBlock):
            x = self.fn(x, emb, mask, cond, **kwargs)
        else:
            x = self.fn(x, **kwargs)
        x = rearrange(x, '(b t) c (w h) -> b c t w h', b=b, c=c, t=t, w=w, h=h)
        return x


class RearrangeTimeChannelCentric(EmbedBlock):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, emb=None, mask=None, cond=None, **kwargs):
        b, c, t, w, h = x.shape
        x = rearrange(x, 'b c t w h -> (b w h) c t')
        if torch.is_tensor(mask):
            mask = rearrange(mask, 'b c t w h -> (b w h) c t')
        if torch.is_tensor(cond):
            cond = rearrange(cond, 'b c t w h -> (b w h) c t')
        if torch.is_tensor(emb):
            emb = rearrange(emb, 'b c t w h -> (b w h) c t')
        if isinstance(self.fn, EmbedBlock):
            x = self.fn(x, emb, mask, cond, **kwargs)
        else:
            x = self.fn(x, **kwargs)
        x = rearrange(x, '(b w h) c t -> b c t w h', b=b, c=c, t=t, w=w, h=h)
        return x
