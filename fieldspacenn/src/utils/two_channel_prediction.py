from typing import Tuple

import torch


def split_mean_and_log_variance(
    prediction: torch.Tensor,
    channel_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return mean and log variance from a two-channel prediction."""
    if prediction.shape[channel_dim] != 2:
        raise ValueError(
            "A mean/log-variance prediction must have exactly two channels, "
            f"got shape {tuple(prediction.shape)} at dimension {channel_dim}."
        )
    return prediction.unbind(dim=channel_dim)


def split_mean_and_error_scale(
    prediction: torch.Tensor,
    channel_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return mean and standard deviation while retaining the channel axis."""
    mean, log_variance = split_mean_and_log_variance(prediction, channel_dim)
    sigma = torch.exp(0.5 * log_variance)
    return mean.unsqueeze(channel_dim), sigma.unsqueeze(channel_dim)
