from typing import Tuple

import torch


MIN_LOG_VARIANCE = -30.0
MAX_LOG_VARIANCE = 20.0


def stabilize_log_variance(log_variance: torch.Tensor) -> torch.Tensor:
    """Bound log variance before converting it to variance or scale."""
    return log_variance.clamp(MIN_LOG_VARIANCE, MAX_LOG_VARIANCE)


def split_mean_and_log_variance(
    prediction: torch.Tensor,
    channel_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return mean and stabilized log variance from a two-channel prediction."""
    if prediction.shape[channel_dim] != 2:
        raise ValueError(
            "A mean/log-variance prediction must have exactly two channels, "
            f"got shape {tuple(prediction.shape)} at dimension {channel_dim}."
        )
    mean, log_variance = prediction.unbind(dim=channel_dim)
    return mean, stabilize_log_variance(log_variance)


def split_mean_and_error_scale(
    prediction: torch.Tensor,
    channel_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return mean and standard deviation while retaining the channel axis."""
    mean, log_variance = split_mean_and_log_variance(prediction, channel_dim)
    sigma = torch.exp(0.5 * log_variance)
    return mean.unsqueeze(channel_dim), sigma.unsqueeze(channel_dim)
