#write a pytorch learning rate scheduler for cosine decay with linear warmup

from torch.optim import Optimizer
import torch
import math
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

class CosineDecayWithLinearWarmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps = 5000, max_steps = 200000, base_lr = 0.0001, final_lr = 1e-6, last_step=-1):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.base_lr = base_lr
        self.final_lr = final_lr
        self.last_step = last_step
        super(CosineDecayWithLinearWarmup, self).__init__(optimizer, last_step)
        
    def get_lr(self):
        if self.last_step < self.warmup_steps:
            #warmup from 0 to base_lr
            return [self.last_step / self.warmup_steps * base_lr for base_lr in self.base_lrs]
        else:
            return [self.final_lr + 0.5 * (base_lr - self.final_lr) * (1 + math.cos(math.pi * (self.last_step - self.warmup_steps) / (self.max_steps - self.warmup_steps))) for base_lr in self.base_lrs]
    
    def step(self, step=None):
        if step is None:
            step = self.last_step + 1
        self.last_step = step
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

# write a dummy test for the scheduler with 5k warmup, 200k max steps, a dummy network and optimizer, and return the step and lr history
    @staticmethod
    def test():
        net = nn.Linear(10, 1)
        optimizer = optim.Adam(net.parameters(), lr=0.0001)
        scheduler = CosineDecayWithLinearWarmup(optimizer, warmup_steps = 5000, max_steps = 200000, base_lr = 0.0001, final_lr = 1e-6)
        step_lr_history = []
        for step in tqdm(range(0, 200000)):
            optimizer.step()
            scheduler.step()
            step_lr_history.append((step, optimizer.param_groups[0]['lr']))
        return step_lr_history