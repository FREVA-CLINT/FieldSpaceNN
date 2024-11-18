from typing import Dict

import torch
import torch.nn as nn


class BaseEmbedder(nn.Module):
    """
    A neural network module to embed longitude and latitude coordinates.

    :param in_channels: Number of input features.
    :param embed_dim: Dimensionality of the embedding output.
    """

    def __init__(self, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.embedding_fn = None

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass to embed the tensor.

        :param emb: Input tensor containing values to be embedded.
        :return: Embedded tensor.
        """
        # Apply the embedder to the input tensor
        return self.embedding_fn(emb)


class EmbedderSequential(nn.Module):
    def __init__(self, embedders: Dict[str, BaseEmbedder], mode='sum'):
        """
        Args:
            embedders (dict): A dictionary of embedders. Keys are names, values are instances of embedders.
            mode (str): Combination mode. Can be 'average', 'sum', or 'concat'.
        """
        super(EmbedderSequential, self).__init__()
        self.embedders = nn.ModuleDict(embedders)
        assert mode in ['average', 'sum', 'concat'], "Mode must be 'average', 'sum', or 'concat'."
        self.mode = mode

    def forward(self, inputs: Dict[str, torch.Tensor], output_shape):
        """
        Args:
            inputs (dict): A dictionary of input tensors where each key corresponds to an embedder.
            output_shape (tuple): The desired output shape for each embedding.

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
            embed_output = embed_output.view(output_shape)
            embeddings.append(embed_output)

        # Combine embeddings according to the mode
        if self.mode == 'concat':
            # Concatenate along the channel dimension
            return torch.cat(embeddings, dim=-1)
        elif self.mode == 'sum':
            # Sum the embeddings
            return torch.stack(embeddings, dim=0).sum(dim=0)
        elif self.mode == 'average':
            # Average the embeddings
            return torch.stack(embeddings, dim=0).mean(dim=0)
