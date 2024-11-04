from abc import abstractmethod

from torch import nn


class EmbedBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb, mask, cond, coords, **kwargs):
        """
        Apply the module to `x` given `emb`, `mask`, `cond`, `coords`.
        """

class EmbedBlockSequential(nn.Sequential, EmbedBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb=None, mask=None, cond=None, coords=None, **kwargs):
        for layer in self:
            if isinstance(layer, EmbedBlock):
                x = layer(x, emb, mask, cond, coords, **kwargs)
            else:
                x = layer(x, **kwargs)
        return x