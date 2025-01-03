import torch
import numpy as np


class AbstractDistribution:
    """
    Abstract base class for probability distributions.
    """

    def sample(self, noise: torch.Tensor = None) -> torch.Tensor:
        """
        Generate a sample from the distribution.
        Raises NotImplementedError if not implemented in subclass.
        """
        raise NotImplementedError()

    def mode(self) -> torch.Tensor:
        """
        Get the mode of the distribution.
        Raises NotImplementedError if not implemented in subclass.
        """
        raise NotImplementedError()


class DiracDistribution(AbstractDistribution):
    """
    Dirac delta distribution, representing a deterministic point value.
    """

    def __init__(self, value: torch.Tensor):
        """
        Initialize the Dirac distribution with a fixed value.

        :param value: Fixed value of the distribution.
        """
        self.value = value

    def sample(self, noise: torch.Tensor = None) -> torch.Tensor:
        """
        Return the fixed value as a sample.
        """
        return self.value

    def mode(self) -> torch.Tensor:
        """
        Return the fixed value as the mode.
        """
        return self.value


class DiagonalGaussianDistribution(AbstractDistribution):
    """
    Diagonal Gaussian distribution, supporting sampling and KL divergence calculations.
    """

    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        """
        Initialize the distribution with mean and log-variance parameters.

        :param parameters: Tensor containing concatenated mean and log-variance values.
        :param deterministic: Boolean flag to determine if the distribution should be deterministic.
        """
        self.parameters = parameters
        # Split parameters into mean and log-variance, clamping log-variance values
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=-1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)  # Standard deviation
        self.var = torch.exp(self.logvar)  # Variance

        # Dimension indices for summing over all dimensions except batch dimension
        self.dims = [i for i in range(1, self.mean.dim())]

        # Override std and var with zeros if deterministic
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self, noise: torch.Tensor = None) -> torch.Tensor:
        """
        Generate a sample from the distribution.

        :param noise: Optional noise tensor for sampling; if None, generates standard Gaussian noise.
        :return: Sampled tensor from the distribution.
        """
        if noise is None:
            noise = torch.randn(self.mean.shape)

        # Sample using reparameterization trick
        x = self.mean + self.std * noise.to(device=self.parameters.device)
        return x

    def kl(self, other: 'DiagonalGaussianDistribution' = None) -> torch.Tensor:
        """
        Compute the KL divergence to another distribution or the standard normal.

        :param other: Another DiagonalGaussianDistribution instance, or None for standard normal.
        :return: KL divergence value as a tensor.
        """
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                # KL divergence to standard normal
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=self.dims
                )
            else:
                # KL divergence to another Gaussian distribution
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=self.dims
                )

    def nll(self, sample: torch.Tensor) -> torch.Tensor:
        """
        Compute the negative log-likelihood of a sample.

        :param sample: Sample tensor for likelihood calculation.
        :return: Negative log-likelihood as a tensor.
        """
        if self.deterministic:
            return torch.Tensor([0.])

        logtwopi = np.log(2.0 * np.pi)
        # Negative log-likelihood computation
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=self.dims
        )

    def mode(self) -> torch.Tensor:
        """
        Get the mode of the distribution (mean for Gaussian).
        """
        return self.mean


def normal_kl(mean1: torch.Tensor, logvar1: torch.Tensor, mean2: torch.Tensor, logvar2: torch.Tensor) -> torch.Tensor:
    """
    Compute the KL divergence between two Gaussian distributions.

    :param mean1: Mean of the first Gaussian.
    :param logvar1: Log-variance of the first Gaussian.
    :param mean2: Mean of the second Gaussian.
    :param logvar2: Log-variance of the second Gaussian.
    :return: KL divergence as a tensor.
    """
    # Determine which argument is a Tensor to set device
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "At least one argument must be a Tensor"

    # Ensure log-variance inputs are Tensors for compatibility with torch.exp()
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    # KL divergence calculation
    return 0.5 * (
            -1.0
            + logvar2
            - logvar1
            + torch.exp(logvar1 - logvar2)
            + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )
