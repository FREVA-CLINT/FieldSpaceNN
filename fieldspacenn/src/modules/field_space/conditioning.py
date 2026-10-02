"""Learned conditioning terms for field-space attention."""

import torch
import torch.nn as nn


class VariableRelativeBias(nn.Module):
    """Per-head attention bias for ordered query/key variable pairs."""

    def __init__(self, n_variables: int, n_heads: int) -> None:
        super().__init__()
        if n_variables <= 0:
            raise ValueError("n_variables must be positive")
        if n_heads <= 0:
            raise ValueError("n_heads must be positive")

        self.n_variables = int(n_variables)
        self.n_heads = int(n_heads)
        self.lookup = nn.Embedding.from_pretrained(
            torch.zeros(self.n_variables**2, self.n_heads),
            freeze=False,
        )

    def forward(
        self,
        query_variable_ids: torch.Tensor,
        key_variable_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Look up biases for all ordered query/key variable pairs.

        :param query_variable_ids: One-dimensional query-variable indices.
        :param key_variable_ids: One-dimensional key-variable indices.
        :return: Bias tensor shaped ``(heads, query_length, key_length)``.
        """
        if query_variable_ids.ndim != 1 or key_variable_ids.ndim != 1:
            raise ValueError("Variable IDs must be one-dimensional")

        pair_ids = (
            query_variable_ids[:, None] * self.n_variables
            + key_variable_ids[None, :]
        )
        return self.lookup(pair_ids).permute(2, 0, 1)
