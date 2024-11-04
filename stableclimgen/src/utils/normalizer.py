import torch
from typing import Union


class DataNormalizer:
    """
    Base class for data normalization, loading statistics from a provided directory.

    :param data_stats_dir: Path to the directory containing data statistics for normalization.
    """

    def __init__(self, data_stats_dir: str):
        self.data_stats = torch.load(data_stats_dir, weights_only=False)

    def normalize(self, data: torch.Tensor, index: int) -> torch.Tensor:
        """
        Normalize the data based on statistics for a specific index.

        :param data: Input data tensor to normalize.
        :param index: Index indicating which data statistics to apply.
        :return: Normalized data tensor.
        """
        pass

    def renormalize(self, data: torch.Tensor, index: int) -> torch.Tensor:
        """
        Reverse the normalization process to return data to its original scale.

        :param data: Normalized data tensor to be renormalized.
        :param index: Index indicating which data statistics to apply.
        :return: Renormalized data tensor.
        """
        pass


class MinMaxNormalizer(DataNormalizer):
    """
    Normalizer that scales data using min-max normalization based on precomputed statistics.

    :param data_stats_dir: Path to the directory containing data statistics for min-max normalization.
    """

    def __init__(self, data_stats_dir: str):
        super().__init__(data_stats_dir)

    def normalize(self, data: torch.Tensor, index: int) -> torch.Tensor:
        """
        Normalize the data using min-max scaling.

        :param data: Input data tensor to normalize.
        :param index: Index indicating which min and max values to use.
        :return: Min-max normalized data tensor.
        """
        return (data - self.data_stats['min'][index]) / self.data_stats['max'][index]

    def renormalize(self, data: torch.Tensor, index: int) -> torch.Tensor:
        """
        Reverse the min-max normalization to return data to its original scale.

        :param data: Min-max normalized data tensor to be renormalized.
        :param index: Index indicating which min and max values to use.
        :return: Renormalized data tensor.
        """
        return data * self.data_stats['max'][index] + self.data_stats['min'][index]