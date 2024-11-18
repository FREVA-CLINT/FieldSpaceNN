import math
import torch
import torch.nn as nn

from stableclimgen.src.modules.embedding.embedder_base import BaseEmbedder


class RandomFourierLayer(nn.Module):
    """
    A neural network layer that applies a Random Fourier Feature transformation.

    :param in_features: Number of input features (default is 3).
    :param n_neurons: Number of neurons in the layer, which will determine the dimensionality of the output (default is 512).
    :param wave_length: Scaling factor for the input tensor, affecting the frequency of the sine and cosine functions (default is 1.0).
    """

    def __init__(
            self,
            in_features: int = 3,
            n_neurons: int = 512,
            wave_length: float = 1.0
    ) -> None:
        super().__init__()
        # Initialize the weights parameter with a random normal distribution
        self.register_parameter(
            "weights",
            torch.nn.Parameter(
                torch.randn(in_features, n_neurons // 2), requires_grad=False
            )
        )
        # Scaling constant to normalize the output
        self.constant = math.sqrt(2 / n_neurons)
        # Wave length for scaling the input tensor
        self.wave_length = wave_length

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the layer, applying Random Fourier Feature transformation.

        :param in_tensor: Input tensor to be transformed.
        :return: Transformed output tensor.
        """
        # Normalize input tensor by the wave length
        in_tensor = in_tensor / self.wave_length
        # Apply a linear transformation using random weights
        out_tensor = 2 * torch.pi * in_tensor @ self.weights
        # Apply sine and cosine functions and concatenate the results
        out_tensor = self.constant * torch.cat(
            [torch.sin(out_tensor), torch.cos(out_tensor)], dim=-1
        )
        return out_tensor


class CoordinateEmbedder(BaseEmbedder):
    """
    A neural network module to embed longitude and latitude coordinates.

    :param embed_dim: Dimensionality of the embedding output.
    :param in_channels: Number of input coordinate features (default is 2).
    """

    def __init__(self, in_channels: int, embed_dim: int) -> None:
        super().__init__(in_channels, embed_dim)
        # Mesh embedder consisting of a RandomFourierLayer followed by linear and GELU activation layers
        self.embedding_fn = torch.nn.Sequential(
            RandomFourierLayer(in_features=self.in_channels, n_neurons=self.embed_dim),
            torch.nn.Linear(self.embed_dim, self.embed_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.embed_dim, self.embed_dim),
        )
