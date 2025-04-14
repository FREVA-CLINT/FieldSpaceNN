import json
import torch
from typing import Dict


class DataNormalizer:
    """
    Base class for data normalization. It loads and uses statistics from a provided dictionary.

    :param stat_dict: Dictionary containing data statistics for normalization (e.g., means, standard deviations, quantiles).
    :param definition_dict: Dictionary containing additional settings for the normalizer (e.g., quantile values, output ranges).
    """

    def __init__(self):
        pass

    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        """
        Abstract method for normalizing the input data tensor.

        :param data: Input data tensor to normalize.
        :return: Normalized data tensor.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def denormalize(self, data: torch.Tensor) -> torch.Tensor:
        """
        Abstract method for denormalizing the input data tensor.

        :param data: Normalized data tensor to be denormalized.
        :return: Denormalized data tensor.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def denormalize_var(self, data_var: torch.Tensor, data=None) -> torch.Tensor:
        """
        Abstract method for denormalizing the variance of the input data tensor.

        :param data: Normalized data tensor to be denormalized.
        :return: Denormalized data tensor.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")


class QuantileNormalizer(DataNormalizer):
    """
    Normalizer that scales data using min-max normalization based on quantiles from precomputed statistics.

    :param stat_dict: Dictionary containing quantile statistics for normalization.
    :param definition_dict: Dictionary containing settings like quantile values and output ranges.
    """

    def __init__(self, stat_dict: Dict, definition_dict: Dict):
        super().__init__()
        
        # Extract quantile settings and output range from the definition dictionary
        quantile = definition_dict['quantile']
        output_range = definition_dict.get('output_range', [0, 1])

        # Calculate the lower and upper quantiles
        q_low = torch.min(torch.tensor([quantile, 1 - quantile]))
        q_high = torch.max(torch.tensor([quantile, 1 - quantile]))

        # Retrieve the actual quantile values from the statistics dictionary
        self.q_low = stat_dict["quantiles"]["{:.2f}".format(float(q_low))]
        self.q_high = stat_dict["quantiles"]["{:.2f}".format(float(q_high))]

        # Set the output range for the normalized data
        self.output_range = output_range

    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        """
        Normalize data using min-max scaling between the specified quantiles.

        :param data: Input data tensor to normalize.
        :return: Min-max normalized data tensor within the specified output range.
        """
        # Apply min-max scaling to map data to the range [0, 1]
        norm_data = (data - self.q_low) / (self.q_high - self.q_low)
        # Scale data to the specified output range
        return norm_data * (self.output_range[1] - self.output_range[0]) + self.output_range[0]

    def denormalize(self, data: torch.Tensor) -> torch.Tensor:
        """
        Reverse the min-max normalization and return data to its original scale.

        :param data: Normalized data tensor to be denormalized.
        :return: Data tensor in the original scale.
        """
        # Rescale data from the output range to [0, 1]
        data_rescaled = (data - self.output_range[0]) / (self.output_range[1] - self.output_range[0])
        # Rescale data to the original min-max range
        return data_rescaled * (self.q_high - self.q_low) + self.q_low

    def denormalize_var(self, data_var: torch.Tensor, data=None) -> torch.Tensor:
        """
        Reverse the min-max normalization and return data to its original scale.

        :param data: Normalized data tensor to be denormalized.
        :return: Data tensor in the original scale.
        """
        # Rescale data from the output range to [0, 1]
        data_rescaled = (data_var) / (self.output_range[1] - self.output_range[0])**2
        # Rescale data to the original min-max range
        return data_rescaled * (self.q_high - self.q_low)**2

class AbsQuantileNormalizer(DataNormalizer):
    """
    Normalizer that scales data using max scaling based on a specified upper quantile from precomputed statistics.

    :param stat_dict: Dictionary containing quantile statistics for normalization.
    :param definition_dict: Dictionary containing settings like quantile values and output ranges.
    """

    def __init__(self, stat_dict: Dict, definition_dict: Dict):
        super().__init__()

        # Extract quantile settings and output range from the definition dictionary
        quantile = definition_dict['quantile']
        output_range = definition_dict.get('output_range', [0, 1])

        # Calculate the upper quantile
        q_high = torch.max(torch.tensor([quantile, 1 - quantile]))

        # Retrieve the actual upper quantile value from the statistics dictionary
        self.q_high = stat_dict["quantiles"]["{:.2f}".format(float(q_high))]

        # Set the output range for the normalized data
        self.output_range = output_range

    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        """
        Normalize data using max scaling relative to the specified upper quantile.

        :param data: Input data tensor to normalize.
        :return: Normalized data tensor within the specified output range.
        """
        # Scale data to the range [0, 1] using the upper quantile
        norm_data = data / self.q_high
        # Scale data to the specified output range
        return norm_data * (self.output_range[1] - self.output_range[0]) + self.output_range[0]

    def denormalize(self, data: torch.Tensor) -> torch.Tensor:
        """
        Reverse the max normalization and return data to its original scale.

        :param data: Normalized data tensor to be denormalized.
        :return: Data tensor in the original scale.
        """
        # Rescale data from the output range to [0, 1]
        data_rescaled = (data - self.output_range[0]) / (self.output_range[1] - self.output_range[0])
        # Rescale data to the original scale using the upper quantile
        return data_rescaled * self.q_high

    def denormalize_var(self, data_var: torch.Tensor, data=None) -> torch.Tensor:
        """
        Reverse the max normalization and return data to its original scale.

        :param data: Normalized data tensor to be denormalized.
        :return: Data tensor in the original scale.
        """
        # Rescale data from the output range to [0, 1]
        data_rescaled = (data) / (self.output_range[1] - self.output_range[0])**2
        # Rescale data to the original scale using the upper quantile
        return data_rescaled * self.q_high**2


class MeanStdNormalizer(DataNormalizer):
    """
    Normalizer that standardizes data using mean and standard deviation from precomputed statistics.

    :param stat_dict: Dictionary containing mean and standard deviation for normalization.
    :param definition_dict: Dictionary containing additional settings for normalization.
    """

    def __init__(self, stat_dict: Dict, definition_dict: Dict):
        super().__init__()

        # Extract mean and standard deviation from the statistics dictionary
        self.mean = stat_dict["mean"]
        self.std = stat_dict["std"]

    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        """
        Standardize data using mean and standard deviation.

        :param data: Input data tensor to normalize.
        :return: Standardized data tensor.
        """
        # Standardize data by subtracting the mean and dividing by the standard deviation
        return (data - self.mean) / self.std

    def denormalize(self, data: torch.Tensor) -> torch.Tensor:
        """
        Reverse the standardization and return data to its original scale.

        :param data: Standardized data tensor to be denormalized.
        :return: Data tensor in the original scale.
        """
        # Rescale data to the original scale by multiplying by std and adding the mean
        return data * self.std + self.mean

    def denormalize_var(self, data_var: torch.Tensor, data=None)-> torch.Tensor:
        """
        Reverse the standardization and return data to its original scale.

        :param data: Standardized data tensor to be denormalized.
        :return: Data tensor in the original scale.
        """
        # Rescale data to the original scale by multiplying by std and adding the mean
        return data * self.std**2
    

class ScaledLogNormalizer(DataNormalizer):
    """
    Normalizer that standardizes data using mean and standard deviation from precomputed statistics.

    :param stat_dict: Dictionary containing mean and standard deviation for normalization.
    :param definition_dict: Dictionary containing additional settings for normalization.
    """

    def __init__(self, stat_dict: Dict, definition_dict: Dict):
        super().__init__()

        self.scale = definition_dict.get('scale', 1e6)
        self.quantile_normalizer = QuantileNormalizer(stat_dict, definition_dict)


    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        """
        Standardize data using mean and standard deviation.

        :param data: Input data tensor to normalize.
        :return: Standardized data tensor.
        """
        # Standardize data by subtracting the mean and dividing by the standard deviation

        data = torch.log1p(data * self.scale)

        return self.quantile_normalizer.normalize(data)

    def denormalize(self, data: torch.Tensor) -> torch.Tensor:
        """
        Reverse the standardization and return data to its original scale.

        :param data: Standardized data tensor to be denormalized.
        :return: Data tensor in the original scale.
        """
        # Rescale data to the original scale by multiplying by std and adding the mean
        data = torch.expm1(data)/self.scale

        return self.quantile_normalizer.denormalize(data)

    def denormalize_var(self, data_var: torch.Tensor, data=None) -> torch.Tensor:
        """
        Reverse the standardization and return data to its original scale.

        :param data: Standardized data tensor to be denormalized.
        :return: Data tensor in the original scale.
        """
        # Rescale data to the original scale by multiplying by std and adding the mean
        return data_var*(torch.expm1(data)/self.scale)**2