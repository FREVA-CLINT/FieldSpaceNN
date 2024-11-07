from abc import abstractmethod
from typing import Optional
import torch
from torch import nn


class EmbedBlock(nn.Module):
    """
    Abstract base module where `forward()` takes timestep embeddings as a second argument.
    This class provides a standard interface for modules that operate with embeddings.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor, emb: Optional[torch.Tensor], mask: Optional[torch.Tensor],
                cond: Optional[torch.Tensor], coords: Optional[torch.Tensor], *args, **kwargs) -> torch.Tensor:
        """
        Apply the module to `x` given `emb`, `mask`, `cond`, `coords`.

        :param x: Input tensor.
        :param emb: Embedding tensor, providing timestep information.
        :param mask: Optional mask tensor.
        :param cond: Optional conditioning tensor.
        :param coords: Optional coordinates tensor.
        :return: Processed tensor after applying the module.
        """
        pass


class EmbedBlockSequential(nn.Sequential, EmbedBlock):
    """
    A sequential module that passes timestep embeddings to child modules that
    support them as an extra input. This class extends `nn.Sequential` to handle
    embedding-based layers within a sequence.
    """

    def forward(self, x: torch.Tensor, emb: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None, cond: Optional[torch.Tensor] = None,
                coords: Optional[torch.Tensor] = None, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass for the EmbedBlockSequential.

        :param x: Input tensor.
        :param emb: Optional embedding tensor to be passed to layers that support it.
        :param mask: Optional mask tensor to be passed to layers that support it.
        :param cond: Optional conditioning tensor to be passed to layers that support it.
        :param coords: Optional coordinates tensor to be passed to layers that support it.
        :return: Output tensor after sequentially applying each layer, with embeddings if supported.
        """
        for layer in self:
            if isinstance(layer, EmbedBlock):
                x = layer(x, emb, mask, cond, coords, *args, **kwargs)
            else:
                x = layer(x, **kwargs)
        return x
