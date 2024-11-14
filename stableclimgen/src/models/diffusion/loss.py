import numpy as np
import torch
from torch import Tensor

def normal_kl(mean1: Tensor, logvar1: Tensor, mean2: Tensor, logvar2: Tensor) -> Tensor:
    """
    Compute the KL divergence between two Gaussian distributions with specified means and log variances.

    This function supports broadcasting, allowing comparisons of batches with scalar values.

    :param mean1: Mean of the first Gaussian distribution (Tensor).
    :param logvar1: Log variance of the first Gaussian distribution (Tensor).
    :param mean2: Mean of the second Gaussian distribution (Tensor).
    :param logvar2: Log variance of the second Gaussian distribution (Tensor).
    :return: KL divergence between the two distributions as a Tensor.
    """
    tensor = None
    # Find the first tensor among the inputs for device compatibility
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "At least one argument must be a Tensor"

    # Ensure variances are Tensors to support broadcasting and exponential calculations
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    # Calculate KL divergence using the formula
    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def approx_standard_normal_cdf(x: Tensor) -> Tensor:
    """
    Approximate the cumulative distribution function (CDF) of the standard normal distribution.

    This approximation uses a fast mathematical approximation for the CDF.

    :param x: Input Tensor to evaluate the CDF at.
    :return: Approximate CDF values as a Tensor.
    """
    # Use an approximation formula involving hyperbolic tangent and cubic term for efficiency
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def continuous_gaussian_log_likelihood(x: Tensor, *, means: Tensor, log_scales: Tensor) -> Tensor:
    """
    Compute the log-likelihood of data points assuming a continuous Gaussian distribution.

    :param x: Target Tensor values (data points).
    :param means: Mean Tensor of the Gaussian distribution.
    :param log_scales: Log standard deviation Tensor of the Gaussian distribution.
    :return: Log-likelihood values as a Tensor in nats (log base e).
    """
    # Center the data by subtracting the mean
    centered_x = x - means
    # Calculate inverse standard deviation for normalizing the data
    inv_stdv = torch.exp(-log_scales)
    # Normalize data by scaling with inverse std deviation
    normalized_x = centered_x * inv_stdv
    # Compute log probability of normalized data under standard normal distribution
    log_probs = torch.distributions.Normal(torch.zeros_like(x), torch.ones_like(x)).log_prob(normalized_x)
    return log_probs