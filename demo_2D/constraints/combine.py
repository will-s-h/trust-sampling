import torch
from .constraint import Constraint

class Combine(Constraint):
    '''
    Constraint where anything within distance r of x and y is valid
    '''
    
    def __init__(self, c1, c2):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
    
    def __str__(self):
        return f'{self.c1}+{self.c2}'
    
    def constraint(self, samples: torch.FloatTensor) -> torch.FloatTensor:
        super()._check_dim(samples)
        return self.relu((samples[..., 0]-self.x) ** 2 + (samples[..., 1]-self.y) ** 2 - self.rsq) * self.slope
    
    def gradient(self, samples: torch.FloatTensor, func = None) -> torch.FloatTensor:
        samples = samples.clone()
        samples[..., 0] -= self.x
        samples[..., 1] -= self.y
        norms = torch.norm(samples, dim=1)
        samples[norms < self.rsq] = 0.0
        return -(samples.T / norms).T 
    
        