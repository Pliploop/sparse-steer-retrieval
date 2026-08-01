from lightning.pytorch.callbacks import Callback
from lightning.pytorch import Trainer
from lightning.pytorch.core import LightningModule

import torch


class BaseCallback(Callback):

    def __init__(self, every_n_steps = 1, every_n_epochs = 1, **kwargs):
        super().__init__(**kwargs)
        self.every_n_steps = every_n_steps
        self.every_n_epochs = every_n_epochs
        
    def _check_step(self, trainer: Trainer, pl_module: LightningModule) -> bool:
        if self.every_n_steps is not None:
            return trainer.global_step % self.every_n_steps == 0
        else:
            return False
        
    def _check_epoch(self, trainer: Trainer, pl_module: LightningModule) -> bool:
        if self.every_n_epochs is not None:
            return trainer.current_epoch % self.every_n_epochs == 0
        else:
            return False

    def _should_run(self, trainer: Trainer, pl_module: LightningModule) -> bool:
        return self._check_step(trainer, pl_module) or self._check_epoch(trainer, pl_module)

    def _should_run_on_validation(self, trainer: Trainer, pl_module: LightningModule) -> bool:
        return self._check_step(trainer, pl_module) or self._check_epoch(trainer, pl_module)

    def _should_run_on_test(self, trainer: Trainer, pl_module: LightningModule) -> bool:
        return self._check_step(trainer, pl_module) or self._check_epoch(trainer, pl_module)


