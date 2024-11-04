import math
import torch
import torch.nn as nn
from torch import Tensor


def diffusion_step_embedding(diffusion_steps: Tensor, dim: int, max_period: int = 10000) -> Tensor:
    """
    Create sinusoidal timestep embeddings for diffusion steps.

    This function generates sinusoidal positional embeddings for each diffusion step,
    where the frequencies are controlled by `max_period`. The embeddings are intended to
    be used in models that benefit from positional information of diffusion steps.

    :param diffusion_steps: A 1-D Tensor of shape [N], where N is the batch size.
                            Each element represents the diffusion step and can be fractional.
    :param dim: Dimensionality of the output embeddings.
    :param max_period: Controls the minimum frequency of the embeddings.
                       Higher values lead to more gradual frequency changes.
    :return: A Tensor of shape [N x dim] containing the positional embeddings for each diffusion step.
    """
    half_dim = dim // 2  # Calculate half of the embedding dimension

    # Create frequencies using exponential decay, controlled by max_period.
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32) / half_dim
    ).to(device=diffusion_steps.device)

    # Calculate arguments for sine and cosine functions
    args = diffusion_steps[:, None].float() * freqs[None]

    # Combine sine and cosine embeddings along the last dimension
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    # Pad with a zero column if `dim` is odd
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

    return embedding


class DiffusionStepEmbedder(nn.Module):
    """
    A neural network module that encodes diffusion steps.

    This class takes as input a sequence of diffusion steps, applies sinusoidal embeddings,
    and then processes these embeddings through a simple feedforward network.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Initializes the DiffusionStepEmbedder module.

        :param in_channels: Number of input channels for the embedding.
        :param out_channels: Number of output channels for the final embedding.
        """
        super().__init__()

        # Define a feedforward network with SiLU activation
        self.embed = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
        )

        self.in_channels = in_channels  # Store the input dimension of the embeddings

    def forward(self, diffusion_steps: Tensor) -> Tensor:
        """
        Forward pass to embed the diffusion steps.

        :param diffusion_steps: A Tensor containing the diffusion steps for each element in the batch.
        :return: A Tensor containing the transformed embeddings after passing through the feedforward network.
        """
        encodings = diffusion_step_embedding(diffusion_steps, self.in_channels)  # Generate sinusoidal embeddings
        return self.embed(encodings)  # Pass the embeddings through the feedforward network
