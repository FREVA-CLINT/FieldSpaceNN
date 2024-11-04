import math

import torch
import torch.nn as nn

def diffusion_step_embedding(diffusion_steps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param diffusion_steps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=diffusion_steps.device)
    args = diffusion_steps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class DiffusionStepEmbedder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
        )
        self.in_channels = in_channels

    def forward(self, diffusion_steps):
        encodings = diffusion_step_embedding(diffusion_steps, self.in_channels)
        return self.embed(encodings)
