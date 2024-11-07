import json
import torch
from typing import Dict


class DataNormalizer:
    """
    Base class for data normalization, loading statistics from a provided JSON file.

    :param data_stats: Path to the JSON file containing data statistics for normalization.
    """

    def __init__(self, data_stats: str):
        with open(data_stats, 'r') as json_file:
            self.data_stats: Dict[str, Dict[int, float]] = json.load(json_file)  # Expected to contain 'min' and 'max' stats

    def normalize(self, data: torch.Tensor, index: int) -> torch.Tensor:
        """
        Normalize the data based on statistics for a specific index.

        :param data: Input data tensor to normalize.
        :param index: Index indicating which data statistics to apply.
        :return: Normalized data tensor.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def denormalize(self, data: torch.Tensor, index: int) -> torch.Tensor:
        """
        Reverse the normalization process to return data to its original scale.

        :param data: Normalized data tensor to be denormalized.
        :param index: Index indicating which data statistics to apply.
        :return: Denormalized data tensor.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")


class MinMaxNormalizer(DataNormalizer):
    """
    Normalizer that scales data using min-max normalization based on precomputed statistics.

    :param data_stats: Path to the JSON file containing min and max values for normalization.
    :param min_range: Minimum range for normalized data. Default is -1.0.
    :param max_range: Maximum range for normalized data. Default is 1.0.
    """

    def __init__(self, data_stats: str, min_range: float = -1.0, max_range: float = 1.0):
        super().__init__(data_stats)
        self.min_range = min_range
        self.max_range = max_range

    def normalize(self, data: torch.Tensor, index: int) -> torch.Tensor:
        """
        Normalize the data using min-max scaling based on precomputed statistics.

        :param data: Input data tensor to normalize.
        :param index: Index indicating which min and max values to use.
        :return: Min-max normalized data tensor within the range [min_range, max_range].
        """
        min_val = self.data_stats['min'][index]
        max_val = self.data_stats['max'][index]

        # Apply min-max scaling to bring data to the range [0, 1]
        norm_data = (data - min_val) / (max_val - min_val)
        # Scale to the specified range [min_range, max_range]
        return norm_data * (self.max_range - self.min_range) + self.min_range

    def denormalize(self, data: torch.Tensor, index: int) -> torch.Tensor:
        """
        Reverse the min-max normalization to return data to its original scale.

        :param data: Min-max normalized data tensor to be denormalized.
        :param index: Index indicating which min and max values to use.
        :return: Denormalized data tensor in the original scale.
        """
        min_val = self.data_stats['min'][index]
        max_val = self.data_stats['max'][index]

        # Scale data back to the range [0, 1]
        data_rescaled = (data - self.min_range) / (self.max_range - self.min_range)
        # Rescale to the original min-max range
        return data_rescaled * (max_val - min_val) + min_val
