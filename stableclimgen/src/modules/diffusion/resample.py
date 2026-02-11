from abc import ABC, abstractmethod
import numpy as np
import torch as th
import torch.distributed as dist
from typing import List, Tuple


def create_named_schedule_sampler(name: str, diffusion_steps: int) -> "ScheduleSampler":
    """
    Create a ScheduleSampler instance based on the specified name.

    :param name: The name of the sampler to create.
    :param diffusion_steps: The number of diffusion steps for the process.
    :return: An instance of a ScheduleSampler subclass.
    """
    if name == "uniform":
        return UniformSampler(diffusion_steps)
    elif name == "loss-second-moment":
        return LossSecondMomentResampler(diffusion_steps)
    else:
        raise NotImplementedError(f"Unknown schedule sampler: {name}")


class ScheduleSampler(ABC):
    """
    Abstract base class representing a distribution over timesteps in the diffusion process.
    Intended to reduce variance in the objective through various sampling strategies.
    """

    @abstractmethod
    def weights(self) -> np.ndarray:
        """
        Obtain an array of weights for each diffusion step.

        :return: A numpy array of positive weights for each diffusion step.
        """

    def sample(self, batch_size: int) -> Tuple[th.Tensor, th.Tensor]:
        """
        Sample timesteps with importance sampling for a batch of specified size.

        :param batch_size: Number of timesteps to sample.
        :return: A tuple containing:
                 - timesteps (torch.Tensor): Tensor of sampled timestep indices of shape ``(b,)``.
                 - weights (torch.Tensor): Tensor of weights corresponding to each sampled index
                   of shape ``(b,)``.
        """
        w = self.weights()  # Retrieve the weights
        p = w / np.sum(w)   # Normalize weights to form a probability distribution
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)  # Sample indices based on probabilities
        indices = th.from_numpy(indices_np).long()  # Convert indices to tensor
        weights_np = 1 / (len(p) * p[indices_np])   # Calculate the importance weights
        weights = th.from_numpy(weights_np).float()  # Convert weights to tensor
        return indices, weights


class UniformSampler(ScheduleSampler):
    """
    Uniform sampler that assigns equal weight to each diffusion step.
    """

    def __init__(self, diffusion_steps: int) -> None:
        """
        Initialize the uniform sampler with the specified diffusion steps.

        :param diffusion_steps: The number of diffusion steps in the process.
        :return: None.
        """
        self.diffusion_steps: int = diffusion_steps
        self._weights: np.ndarray = np.ones([diffusion_steps])  # Equal weight for each step

    def weights(self) -> np.ndarray:
        """
        Return uniform weights for each diffusion step.

        :return: A numpy array of equal weights for each diffusion step.
        """
        return self._weights


class LossAwareSampler(ScheduleSampler):
    """
    Base class for samplers that update weights based on model losses at each timestep.
    """

    def update_with_local_losses(self, local_ts: th.Tensor, local_losses: th.Tensor) -> None:
        """
        Synchronize weights across distributed nodes based on local timestep losses.

        :param local_ts: Tensor of sampled timesteps of shape ``(b,)``.
        :param local_losses: Tensor of losses corresponding to each timestep of shape ``(b,)``.
        :return: None.
        """
        batch_sizes = [th.tensor([0], dtype=th.int32, device=local_ts.device)
                       for _ in range(dist.get_world_size())]
        dist.all_gather(batch_sizes, th.tensor([len(local_ts)], dtype=th.int32, device=local_ts.device))

        # Determine the maximum batch size and pad if needed
        batch_sizes = [x.item() for x in batch_sizes]
        max_bs = max(batch_sizes)

        # Gather all timesteps and losses from distributed nodes
        timestep_batches = [th.zeros(max_bs).to(local_ts) for bs in batch_sizes]
        loss_batches = [th.zeros(max_bs).to(local_losses) for bs in batch_sizes]
        dist.all_gather(timestep_batches, local_ts)
        dist.all_gather(loss_batches, local_losses)

        # Extract timesteps and losses from gathered data
        timesteps = [x.item() for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]]
        losses = [x.item() for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]]
        self.update_with_all_losses(timesteps, losses)

    @abstractmethod
    def update_with_all_losses(self, ts: List[int], losses: List[float]) -> None:
        """
        Update the reweighting using losses from a model.

        :param ts: List of timesteps as integers.
        :param losses: List of losses corresponding to each timestep.
        :return: None.
        """


class LossSecondMomentResampler(LossAwareSampler):
    """
    Sampler that reweights diffusion steps based on the second moment of past losses.
    """

    def __init__(self, diffusion_steps: int, history_per_term: int = 10, uniform_prob: float = 0.001) -> None:
        """
        Initialize with diffusion steps, history per term, and uniform probability.

        :param diffusion_steps: Total number of diffusion steps.
        :param history_per_term: Number of past loss values to store per diffusion step.
        :param uniform_prob: Probability for uniform sampling as a regularization factor.
        :return: None.
        """
        self.diffusion_steps: int = diffusion_steps
        self.history_per_term: int = history_per_term
        self.uniform_prob: float = uniform_prob
        self._loss_history: np.ndarray = np.zeros([diffusion_steps, history_per_term], dtype=np.float64)
        self._loss_counts: np.ndarray = np.zeros([diffusion_steps], dtype=np.int32)

    def weights(self) -> np.ndarray:
        """
        Calculate the reweighting based on the second moment of past losses.

        :return: Numpy array of weights for each diffusion step.
        """
        if not self._warmed_up():
            return np.ones([self.diffusion_steps], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))  # Calculate second moment
        weights /= np.sum(weights)  # Normalize
        weights *= 1 - self.uniform_prob  # Apply uniform probability regularization
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts: List[int], losses: List[float]) -> None:
        """
        Update the loss history and reweighting for each timestep.

        :param ts: List of timesteps.
        :param losses: List of losses corresponding to each timestep.
        :return: None.
        """
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                # Replace oldest loss if history limit reached
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self) -> bool:
        """
        Check if each timestep has sufficient loss history for accurate weighting.

        :return: Boolean indicating if the sampler has sufficient history.
        """
        return (self._loss_counts == self.history_per_term).all()
