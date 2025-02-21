import sys
from typing import Dict

import torch
import torch.nn as nn
from torch import ModuleDict

from .embedding_layers import RandomFourierLayer, SinusoidalLayer
from ...utils.helpers import expand_tensor


class BaseEmbedder(nn.Module):
    """
    A neural network module to embed longitude and latitude coordinates.

    :param in_channels: Number of input features.
    :param embed_dim: Dimensionality of the embedding output.
    """

    def __init__(self, name: str, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.name = name
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.embedding_fn = None

        self.keep_dims = []

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass to embed the tensor.

        :param emb: Input tensor containing values to be embedded.
        :return: Embedded tensor.
        """
        # Apply the embedder to the input tensor
        return self.embedding_fn(emb)


class CoordinateEmbedder(BaseEmbedder):
    """
    A neural network module to embed longitude and latitude coordinates.

    :param embed_dim: Dimensionality of the embedding output.
    :param in_channels: Number of input coordinate features (default is 2).
    """

    def __init__(self, name: str, in_channels: int, embed_dim: int, wave_length: float=1.0) -> None:
        super().__init__(name, in_channels, embed_dim)

        # keep batch, spatial, variable and channel dimensions
        self.keep_dims = ["b", "s", "v", "c"]

        # Mesh embedder consisting of a RandomFourierLayer followed by linear and GELU activation layers
        self.embedding_fn = torch.nn.Sequential(
            RandomFourierLayer(in_features=self.in_channels, n_neurons=self.embed_dim, wave_length=wave_length),
            torch.nn.Linear(self.embed_dim, self.embed_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.embed_dim, self.embed_dim),
        )

class MaskEmbedder(BaseEmbedder):
    """
    A neural network module to embed longitude and latitude coordinates.

    :param embed_dim: Dimensionality of the embedding output.
    :param in_channels: Number of input coordinate features (default is 2).
    """

    def __init__(self, name: str, in_channels: int, embed_dim: int) -> None:
        super().__init__(name, in_channels, embed_dim)

        # keep batch, spatial, variable and channel dimensions
        self.keep_dims = ["b", "t", "s", "v", "c"]

        # Mesh embedder consisting of a RandomFourierLayer followed by linear and GELU activation layers
        self.embedding_fn = torch.nn.Sequential(
            torch.nn.Linear(self.in_channels, self.embed_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.embed_dim, self.embed_dim),
        )

class VariableEmbedder(BaseEmbedder):

    def __init__(self, name: str, in_channels: int, embed_dim: int, init_value:float = None) -> None:
        super().__init__(name, in_channels, embed_dim)

        self.keep_dims = ["b", "v", "c"]

        self.embedding_fn = nn.Embedding(self.in_channels, self.embed_dim)

        if init_value is not None:
            self.embedding_fn.weight.data.fill_(init_value)

class GridEmbedder(BaseEmbedder):

    def __init__(self, name: str, in_channels: int, embed_dim: int, init_value:float = None) -> None:
        super().__init__(name, in_channels, embed_dim)

        self.keep_dims = ["c"]

        self.embedding_fn = nn.Embedding(self.in_channels, self.embed_dim)

        if init_value is not None:
            self.embedding_fn.weight.data.fill_(init_value)


class DiffusionStepEmbedder(BaseEmbedder):
    """
    A neural network module that encodes diffusion steps.

    This class takes as input a sequence of diffusion steps, applies sinusoidal embeddings,
    and then processes these embeddings through a simple feedforward network.
    """

    def __init__(self, name: str, in_channels: int, embed_dim: int):
        """
        Initializes the DiffusionStepEmbedder module.

        :param in_channels: Number of input channels for the embedding.
        :param embed_dim: Number of output channels for the final embedding.
        """
        super().__init__(name, in_channels, embed_dim)
        # keep batch and channel dimensions
        self.keep_dims = ["b", "c"]

        # Define a feedforward network with SiLU activation
        self.embedding_fn = nn.Sequential(
            SinusoidalLayer(in_channels),
            nn.Linear(self.in_channels, self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )


# Embedder manager to handle shared or non-shared instances
class EmbedderManager:
    _instance = None
    _initialized = False
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(EmbedderManager, cls).__new__(
                cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.shared_embedders = {}
            self._initialized = True

    def get_embedder(self, name, in_channels=None, embed_dim=None, shared=True, **kwargs):
        current_module = sys.modules[__name__]

        # Use getattr to get the class from the current module
        embedder_class = getattr(current_module, name)
        if shared:
            if name not in self.shared_embedders.keys():
                self.shared_embedders[name] = embedder_class(name, in_channels, embed_dim, **kwargs)
            return self.shared_embedders[name]
        else:
            # Create a new instance each time
            return embedder_class(name, in_channels, embed_dim, **kwargs)


class EmbedderSequential(nn.Module):
    def __init__(self, embedders: ModuleDict, mode='sum', spatial_dim_count = 2):
        """
        Args:
            embedders (dict): A dictionary of embedders. Keys are names, values are instances of embedders.
            mode (str): Combination mode. Can be 'average', 'sum', or 'concat'.
        """
        super(EmbedderSequential, self).__init__()
        self.embedders = embedders
        assert mode in ['average', 'sum', 'concat'], "Mode must be 'average', 'sum', or 'concat'."
        self.mode = mode
        self.spatial_dim_count = spatial_dim_count
        self.activation = nn.SiLU()

    def forward(self, inputs: Dict[str, torch.Tensor]):
        """
        Args:
            inputs (dict): A dictionary of input tensors where each key corresponds to an embedder.

        Returns:
            torch.Tensor: The combined embedding tensor.
        """
        embeddings = []

        # Apply each embedder to its respective input
        for embedder_name, embedder in self.embedders.items():
            # Get the input tensor for the current embedder
            if embedder_name not in inputs:
                raise ValueError(f"Input for embedder '{embedder_name}' is missing.")

            input_tensor = inputs[embedder_name]
            embed_output = embedder(input_tensor)

            # Reshape the output to the target output_shape
            embed_output = expand_tensor(embed_output, dims=4 + self.spatial_dim_count, keep_dims=embedder.keep_dims)
            embeddings.append(embed_output)

        # Combine embeddings according to the mode
        if self.mode == 'concat':
            # Concatenate along the channel dimension
            embed_out = torch.cat(embeddings, dim=-1)
        elif self.mode == 'sum':
            # Sum the embeddings
            emb_sum = embeddings[0]
            for emb in embeddings[1:]:
                emb_sum = emb_sum + emb
            embed_out = emb_sum
        elif self.mode == 'average':
            # Sum the embeddings
            emb_sum = embeddings[0]
            for emb in embeddings[1:]:
                emb_sum = emb_sum + emb
            embed_out = emb_sum / (len(embeddings) + 1)
        return self.activation(embed_out)

    @property
    def get_out_channels(self):
        if self.mode == "concat":
            return sum([emb.embed_dim for _, emb in self.embedders.items()])
        else:
            return [emb.embed_dim for _, emb in self.embedders.items()][-1]