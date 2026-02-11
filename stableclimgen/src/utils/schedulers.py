import math
from typing import List, Sequence

import torch


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_iters: int,
        iter_start: int = 0,
    ):
        """
        Initialize the cosine scheduler with optional warmup and zero-iteration phases.

        :param optimizer: Optimizer whose learning rate will be scheduled.
        :param max_iters: Number of iterations for one cosine cycle.
        :param iter_start: Starting iteration offset.
        :return: None.
        """
        self.max_num_iters: int = max_iters
        self.iter_start: int = iter_start

        # Fetch per-group warmup and zero_iters from optimizer.param_groups
        self.warmups: Sequence[int] = [group.get("warmup", 1) for group in optimizer.param_groups]
        self.zero_iters: Sequence[int] = [group.get("zero_iters", 0) for group in optimizer.param_groups]

        super().__init__(optimizer)

    def get_lr(self):
        """
        Compute learning rates for the current iteration.

        :return: List of learning rates per parameter group.
        """
        factor = self.get_lr_factors(self.last_epoch)
        return [base_lr * f for base_lr, f in zip(self.base_lrs, factor)]

    def get_lr_factors(self, epoch: int):
        """
        Compute multiplicative LR factors for each parameter group.

        :param epoch: Current epoch/iteration index.
        :return: List of LR factors per parameter group.
        """
        epoch += self.iter_start
        lr_factors = [0.5 * (1 + math.cos(math.pi * epoch / self.max_num_iters)) for _ in self.optimizer.param_groups]

        for i in range(len(lr_factors)):
            if epoch < self.zero_iters[i]:
                lr_factors[i] = 0.0
            elif epoch <= self.warmups[i] and self.warmups[i] > 0:
                lr_factors[i] *= epoch / self.warmups[i]
            elif epoch <= self.warmups[i] and self.warmups[i] == 0:
                lr_factors[i] *= epoch

        return lr_factors
    

class LinearWarmUpScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Linear warm-up scheduler for learning rates, interpolating between base and target rates.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, warm_up_iters: int, base_lr: float, target_lr: float, last_epoch: int = -1):
        """
        Initialize the linear warm-up scheduler.

        :param optimizer: Optimizer whose learning rate needs to be scheduled.
        :param warm_up_iters: Number of iterations for linear warm-up.
        :param base_lr: Starting learning rate.
        :param target_lr: Target learning rate after warm-up.
        :param last_epoch: Last epoch for the scheduler.
        """
        self.warm_up_iters: int = warm_up_iters
        self.base_lr: float = base_lr
        self.target_lr: float = target_lr
        super(LinearWarmUpScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Compute the learning rate for the current epoch based on the warm-up schedule.

        :return: List of learning rates for each parameter group in the optimizer.
        """
        if self.last_epoch > self.warm_up_iters:
            raise ValueError("Warm-up phase is complete")

        # Linear interpolation between base and target learning rate
        lr = [
            self.base_lr + (self.target_lr - self.base_lr) * (self.last_epoch / self.warm_up_iters)
            for base_lr in self.base_lrs
        ]
        return lr
