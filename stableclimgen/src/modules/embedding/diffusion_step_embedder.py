import math
import torch
import torch.nn as nn

from stableclimgen.src.modules.embedding.embedder_base import BaseEmbedder


class SinusoidalLayer(nn.Module):
    """
    Sinusoidal timestep embeddings for diffusion steps.

    This function generates sinusoidal positional embeddings for each diffusion step,
    where the frequencies are controlled by `max_period`. The embeddings are intended to
    be used in models that benefit from positional information of diffusion steps.

    :param in_channels: Number of input features.
    :param dim: Dimensionality of the output embeddings.
    :param max_period: Controls the minimum frequency of the embeddings.
                       Higher values lead to more gradual frequency changes.
    """
    def __init__(self, in_channels: int, max_period: int = 10000):
        super().__init__()
        self.in_channels = in_channels
        self.freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=in_channels // 2,
                                                 dtype=torch.float32) / (in_channels // 2)
        )

    def forward(self, diffusion_steps):
        """
        :param diffusion_steps: A 1-D Tensor of shape [N], where N is the batch size.
                        Each element represents the diffusion step and can be fractional.
        :return: A Tensor of shape [N x dim] containing the positional embeddings for each diffusion step.
        """

        # Calculate arguments for sine and cosine functions
        args = diffusion_steps[:, None].float() * self.freqs[None].to(diffusion_steps.device)

        # Combine sine and cosine embeddings along the last dimension
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        # Pad with a zero column if `dim` is odd
        if self.in_channels % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

        return embedding


class DiffusionStepEmbedder(BaseEmbedder):
    """
    A neural network module that encodes diffusion steps.

    This class takes as input a sequence of diffusion steps, applies sinusoidal embeddings,
    and then processes these embeddings through a simple feedforward network.
    """

    def __init__(self, in_channels: int, embed_dim: int):
        """
        Initializes the DiffusionStepEmbedder module.

        :param in_channels: Number of input channels for the embedding.
        :param embed_dim: Number of output channels for the final embedding.
        """
        super().__init__(in_channels, embed_dim)

        # Define a feedforward network with SiLU activation
        self.embedding_fn = nn.Sequential(
            SinusoidalLayer(in_channels),
            nn.Linear(self.in_channels, self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )