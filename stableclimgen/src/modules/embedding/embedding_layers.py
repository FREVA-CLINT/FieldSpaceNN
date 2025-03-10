import math
import torch
import torch.nn as nn


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
            wave_length: float = 1.0,
            wave_length_2: float = None
    ) -> None:
        super().__init__()
        # Initialize the weights parameter with a random normal distribution

        if wave_length_2 is None:
            wave_length_2 = wave_length
  
        weights = torch.concat((
            torch.randn(in_features, n_neurons // 4)/ wave_length,
            torch.randn(in_features, n_neurons // 4)/ wave_length_2),dim=1)

        self.register_parameter(
            "weights",
            torch.nn.Parameter(
                weights, requires_grad=False
            )
        )
        # Scaling constant to normalize the output
        self.constant = math.sqrt(2 / n_neurons)

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the layer, applying Random Fourier Feature transformation.

        :param in_tensor: Input tensor to be transformed.
        :return: Transformed output tensor.
        """
        # Normalize input tensor by the wave length
        in_tensor = in_tensor #/ self.wave_length
        # Apply a linear transformation using random weights
        out_tensor = 2 * torch.pi * in_tensor @ self.weights
        # Apply sine and cosine functions and concatenate the results
        out_tensor = self.constant * torch.cat(
            [torch.sin(out_tensor), torch.cos(out_tensor)], dim=-1
        )
        return out_tensor


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
