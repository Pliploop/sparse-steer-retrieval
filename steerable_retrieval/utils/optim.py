import abc
from typing import Tuple

import torch


class LazyOptimizer(abc.ABC):
    r"""Lazy implementation of optimizers. Contrary to standard PyTorch optimizers, these don't require the network
    parameters at initialization and can therefore be configured directly from the command line then instantiated
    properly afterwards.
    """
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.optimizer_class = None

    def __call__(self, parameters) -> torch.optim.Optimizer:
        return self.optimizer_class(parameters, *self.args, **self.kwargs)

    def __str__(self):
        params = '\n'.join(f"\t{arg}," for arg in self.args) + '\n'.join(f"\t{k}: {v}" for k, v in self.kwargs.items())
        return self.optimizer_class.__name__ + "(\n" + params + "\n)"


class Adam(LazyOptimizer):
    def __init__(
            self,
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 0.,
            amsgrad: bool = False,
            **kwargs
    ):
        super(Adam, self).__init__(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            **kwargs
        )
        self.optimizer_class = torch.optim.Adam


class LazyScheduler(abc.ABC):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.scheduler_class = None

    def __call__(self, optimizer):
        return self.scheduler_class(optimizer, *self.args, **self.kwargs)

    def __str__(self):
        params = '\n'.join(f"\t{arg}," for arg in self.args) + '\n'.join(f"\t{k}: {v}" for k, v in self.kwargs.items())
        return self.scheduler_class.__name__ + "(\n" + params + "\n)"


class CosineAnnealing(LazyScheduler):
    def __init__(
            self,
            T_max: int,
            eta_min: float = 0,
            last_epoch: int = -1,
            verbose: bool = False
    ):
        super(CosineAnnealing, self).__init__(
            T_max=T_max,
            eta_min=eta_min,
            last_epoch=last_epoch,
            verbose=verbose
        )
        self.scheduler_class = torch.optim.lr_scheduler.CosineAnnealingLR
