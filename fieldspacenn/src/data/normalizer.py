import json
from typing import Any, Dict, Optional

import torch


def _to_stat_tensor(value: Any) -> torch.Tensor:
    """Convert scalar or list statistics to a tensor for broadcasting."""
    return torch.as_tensor(value, dtype=torch.float32)


def _match_stat_shape(stat: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
    """
    Broadcast scalar stats directly and 1D stats across the detected level axis.

    Per-level statistics are expected to match either:
    - axis 1 for raw dataset tensors shaped like ``(t, level, ...)``
    - axis -2 for model/output tensors shaped like ``(..., level, feature)``
    """
    stat = stat.to(device=data.device, dtype=data.dtype)
    if stat.ndim == 0:
        return stat
    if stat.ndim != 1:
        raise ValueError(f"Expected scalar or 1D statistics, got shape {tuple(stat.shape)}")

    n_levels = stat.shape[0]
    level_axis = None
    if data.ndim > 1 and data.shape[1] == n_levels:
        level_axis = 1
    elif data.ndim > 1 and data.shape[-2] == n_levels:
        level_axis = data.ndim - 2

    if level_axis is None:
        raise ValueError(
            f"Could not align {n_levels} per-level statistics with data shape {tuple(data.shape)}"
        )

    view_shape = [1] * data.ndim
    view_shape[level_axis] = n_levels
    return stat.view(*view_shape)


class DataNormalizer:
    """
    Base class for data normalization. It loads and uses statistics from a provided dictionary.

    :param stat_dict: Dictionary containing data statistics for normalization (e.g., means, standard deviations, quantiles).
    :param definition_dict: Dictionary containing additional settings for the normalizer (e.g., quantile values, output ranges).
    """

    def __init__(self):
        pass

    def normalize(self, data: torch.Tensor):
        """
        Abstract method for normalizing the input data tensor.

        :param data: Input data tensor of shape ``(b, v, t, n, d, f)``.
        :return: Normalized data tensor with the same shape.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def denormalize(self, data: torch.Tensor):
        """
        Abstract method for denormalizing the input data tensor.

        :param data: Normalized data tensor of shape ``(b, v, t, n, d, f)``.
        :return: Denormalized data tensor with the same shape.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def denormalize_var(self, data_var: torch.Tensor, data: Optional[torch.Tensor] = None):
        """
        Abstract method for denormalizing the variance of the input data tensor.

        :param data_var: Normalized variance tensor of shape ``(b, v, t, n, d, f)``.
        :param data: Optional normalized data tensor for scale-dependent variance.
        :return: Denormalized variance tensor with the same shape.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")


class QuantileNormalizer(DataNormalizer):
    """
    Normalizer that scales data using min-max normalization based on quantiles from precomputed statistics.

    :param stat_dict: Dictionary containing quantile statistics for normalization.
    :param definition_dict: Dictionary containing settings like quantile values and output ranges.
    """

    def __init__(self, stat_dict: Dict[str, Any], definition_dict: Dict[str, Any]):
        super().__init__()
        
        # Extract quantile settings and output range from the definition dictionary
        quantile = definition_dict['quantile']
        output_range = definition_dict.get('output_range', [0, 1])

        # Calculate the lower and upper quantiles
        q_low = torch.min(torch.tensor([quantile, 1 - quantile]))
        q_high = torch.max(torch.tensor([quantile, 1 - quantile]))

        # Retrieve the actual quantile values from the statistics dictionary
        self.q_low: torch.Tensor = _to_stat_tensor(stat_dict["quantiles"]["{:.2f}".format(float(q_low))])
        self.q_high: torch.Tensor = _to_stat_tensor(stat_dict["quantiles"]["{:.2f}".format(float(q_high))])

        # Set the output range for the normalized data
        self.output_range: list = output_range

    def normalize(self, data: torch.Tensor):
        """
        Normalize data using min-max scaling between the specified quantiles.

        :param data: Input data tensor of shape ``(b, v, t, n, d, f)``.
        :return: Min-max normalized data tensor within the specified output range.
        """
        # Apply min-max scaling to map data to the range [0, 1]
        q_low = _match_stat_shape(self.q_low, data)
        q_high = _match_stat_shape(self.q_high, data)
        norm_data = (data - q_low) / (q_high - q_low)
        # Scale data to the specified output range
        return norm_data * (self.output_range[1] - self.output_range[0]) + self.output_range[0]

    def denormalize(self, data: torch.Tensor):
        """
        Reverse the min-max normalization and return data to its original scale.

        :param data: Normalized data tensor of shape ``(b, v, t, n, d, f)``.
        :return: Data tensor in the original scale.
        """
        # Rescale data from the output range to [0, 1]
        data_rescaled = (data - self.output_range[0]) / (self.output_range[1] - self.output_range[0])
        # Rescale data to the original min-max range
        q_low = _match_stat_shape(self.q_low, data)
        q_high = _match_stat_shape(self.q_high, data)
        return data_rescaled * (q_high - q_low) + q_low

    def denormalize_var(self, data_var: torch.Tensor, data: Optional[torch.Tensor] = None):
        """
        Reverse the min-max normalization and return data to its original scale.

        :param data_var: Normalized variance tensor of shape ``(b, v, t, n, d, f)``.
        :param data: Optional normalized data tensor (unused).
        :return: Denormalized variance tensor.
        """
        # Rescale data from the output range to [0, 1]
        data_rescaled = (data_var) / (self.output_range[1] - self.output_range[0])**2
        # Rescale data to the original min-max range
        q_low = _match_stat_shape(self.q_low, data_var)
        q_high = _match_stat_shape(self.q_high, data_var)
        return data_rescaled * (q_high - q_low)**2

class AbsQuantileNormalizer(DataNormalizer):
    """
    Normalizer that scales data using max scaling based on a specified upper quantile from precomputed statistics.

    :param stat_dict: Dictionary containing quantile statistics for normalization.
    :param definition_dict: Dictionary containing settings like quantile values and output ranges.
    """

    def __init__(self, stat_dict: Dict[str, Any], definition_dict: Dict[str, Any]):
        super().__init__()

        # Extract quantile settings and output range from the definition dictionary
        quantile = definition_dict['quantile']
        output_range = definition_dict.get('output_range', [0, 1])

        # Calculate the upper quantile
        q_high = torch.max(torch.tensor([quantile, 1 - quantile]))

        # Retrieve the actual upper quantile value from the statistics dictionary
        self.q_high: torch.Tensor = _to_stat_tensor(stat_dict["quantiles"]["{:.2f}".format(float(q_high))])

        # Set the output range for the normalized data
        self.output_range: list = output_range

    def normalize(self, data: torch.Tensor):
        """
        Normalize data using max scaling relative to the specified upper quantile.

        :param data: Input data tensor of shape ``(b, v, t, n, d, f)``.
        :return: Normalized data tensor within the specified output range.
        """
        # Scale data to the range [0, 1] using the upper quantile
        q_high = _match_stat_shape(self.q_high, data)
        norm_data = data / q_high
        # Scale data to the specified output range
        return norm_data * (self.output_range[1] - self.output_range[0]) + self.output_range[0]

    def denormalize(self, data: torch.Tensor):
        """
        Reverse the max normalization and return data to its original scale.

        :param data: Normalized data tensor of shape ``(b, v, t, n, d, f)``.
        :return: Data tensor in the original scale.
        """
        # Rescale data from the output range to [0, 1]
        data_rescaled = (data - self.output_range[0]) / (self.output_range[1] - self.output_range[0])
        # Rescale data to the original scale using the upper quantile
        q_high = _match_stat_shape(self.q_high, data)
        return data_rescaled * q_high

    def denormalize_var(self, data_var: torch.Tensor, data: Optional[torch.Tensor] = None):
        """
        Reverse the max normalization and return data to its original scale.

        :param data_var: Normalized variance tensor of shape ``(b, v, t, n, d, f)``.
        :param data: Optional normalized data tensor (unused).
        :return: Denormalized variance tensor.
        """
        # Rescale data from the output range to [0, 1]
        data_rescaled = data_var / (self.output_range[1] - self.output_range[0])**2
        # Rescale data to the original scale using the upper quantile
        q_high = _match_stat_shape(self.q_high, data_var)
        return data_rescaled * q_high**2


class MeanStdNormalizer(DataNormalizer):
    """
    Normalizer that standardizes data using mean and standard deviation from precomputed statistics.

    :param stat_dict: Dictionary containing mean and standard deviation for normalization.
    :param definition_dict: Dictionary containing additional settings for normalization.
    """

    def __init__(self, stat_dict: Dict[str, Any], definition_dict: Dict[str, Any]):
        super().__init__()

        # Extract mean and standard deviation from the statistics dictionary
        self.mean: torch.Tensor = _to_stat_tensor(stat_dict["mean"])
        self.std: torch.Tensor = _to_stat_tensor(stat_dict["std"])

    def normalize(self, data: torch.Tensor):
        """
        Standardize data using mean and standard deviation.

        :param data: Input data tensor of shape ``(b, v, t, n, d, f)``.
        :return: Standardized data tensor.
        """
        # Standardize data by subtracting the mean and dividing by the standard deviation
        mean = _match_stat_shape(self.mean, data)
        std = _match_stat_shape(self.std, data)
        return (data - mean) / std

    def denormalize(self, data: torch.Tensor):
        """
        Reverse the standardization and return data to its original scale.

        :param data: Standardized data tensor of shape ``(b, v, t, n, d, f)``.
        :return: Data tensor in the original scale.
        """
        # Rescale data to the original scale by multiplying by std and adding the mean
        mean = _match_stat_shape(self.mean, data)
        std = _match_stat_shape(self.std, data)
        return data * std + mean

    def denormalize_var(self, data_var: torch.Tensor, data: Optional[torch.Tensor] = None):
        """
        Reverse the standardization and return data to its original scale.

        :param data_var: Standardized variance tensor of shape ``(b, v, t, n, d, f)``.
        :param data: Optional standardized data tensor (unused).
        :return: Denormalized variance tensor.
        """
        # Rescale data to the original scale by multiplying by std and adding the mean
        std = _match_stat_shape(self.std, data_var)
        return data_var * std**2
    

class ScaledLogNormalizer(DataNormalizer):
    """
    Normalizer that standardizes data using mean and standard deviation from precomputed statistics.

    :param stat_dict: Dictionary containing mean and standard deviation for normalization.
    :param definition_dict: Dictionary containing additional settings for normalization.
    """

    def __init__(self, stat_dict: Dict[str, Any], definition_dict: Dict[str, Any]):
        super().__init__()

        self.scale: float = definition_dict.get('scale', 1e6)

        stat_dict_ = stat_dict.copy()
        stat_dict_['quantiles'] = stat_dict['quantiles'].copy()

        for key, val in stat_dict['quantiles'].items():
            stat_dict_['quantiles'][key] = float(torch.log1p(torch.tensor(val * self.scale)))

        self.quantile_normalizer = QuantileNormalizer(stat_dict_, definition_dict)
        self.zero_offset = float(
            self.quantile_normalizer.normalize(torch.tensor(0.0))
        )


    def normalize(self, data: torch.Tensor):
        """
        Standardize data using mean and standard deviation.

        :param data: Input data tensor of shape ``(b, v, t, n, d, f)``.
        :return: Standardized data tensor.
        """
        # Standardize data by subtracting the mean and dividing by the standard deviation

        data = torch.log1p(data * self.scale)

        return self.quantile_normalizer.normalize(data) - self.zero_offset

    def denormalize(self, data: torch.Tensor):
        """
        Reverse the standardization and return data to its original scale.

        :param data: Standardized data tensor of shape ``(b, v, t, n, d, f)``.
        :return: Data tensor in the original scale.
        """
        data = self.quantile_normalizer.denormalize(data + self.zero_offset)
        # Rescale data to the original scale by multiplying by std and adding the mean
        data = torch.expm1(data)/self.scale

        return data

    def denormalize_var(self, data_var: torch.Tensor, data: Optional[torch.Tensor] = None):
        """
        Reverse the standardization and return data to its original scale.

        :param data_var: Standardized variance tensor of shape ``(b, v, t, n, d, f)``.
        :param data: Standardized data tensor used for scale.
        :return: Denormalized variance tensor.
        """
        # Rescale data to the original scale by multiplying by std and adding the mean
        return data_var*(torch.expm1(data)/self.scale)**2
