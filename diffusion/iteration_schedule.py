import numpy as np
import random

## Iterations Max Schedulers ##

class LinearScheduler:
    def __init__(self, start=1, end=5, steps=1000):
        self.steps = steps
        self.start, self.end = start, end
        self.maxes = np.clip(np.round(np.linspace(end+0.5, start-0.5, steps)), start, end)

    def __call__(self, time):
        assert time >= 0 and time < self.steps
        return self.maxes[time]
    
    def __str__(self):
        return f"LinearSchedule_{self.start}to{self.end}"

class InverseScheduler:
    def __init__(self, betas=None, total_steps=1000, ddim_steps=200, nfes=1000):
        self.steps = total_steps
        self.betas = betas if betas is not None else np.linspace(1e-4, 2e-2, total_steps, dtype=np.float64)
        times = np.linspace(-1, total_steps - 1, ddim_steps + 1)[1:].astype(int)
        self.maxes = (nfes-ddim_steps) / (self.betas * np.sum(1 / self.betas[times[1:]]))
    
    def _random_round(val):
        low = int(val)
        return low if (low + random.random() < val) else low + 1

    def __call__(self, time):
        assert time >= 0 and time < self.steps
        return InverseScheduler._random_round(self.maxes[time])
    
    def __str__(self):
        return "InverseScheduler"
    
    
## Norm Schedulers ##

class InverseNormScheduler:
    def __init__(self, J_scheduler, base_norm=1):
        self.base_norm = base_norm
        self.J_scheduler = J_scheduler

    def __call__(self, time):
        return self.base_norm / (self.J_scheduler.maxes[time] ** 0.5)

    def __str__(self):
        return f"InverseNormScheduler{self.base_norm}"
        
        