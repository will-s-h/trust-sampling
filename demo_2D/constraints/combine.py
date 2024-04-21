import torch
from .constraint import Constraint
from .circle import CircleInequality

class Combine(Constraint):
    '''
    Combine two constraints by adding their gradients
    '''
    
    def __init__(self, c1: Constraint, c2: Constraint):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
    
    def __str__(self):
        return f'{self.c1}+{self.c2}'
    
    def constraint(self, samples: torch.FloatTensor) -> torch.FloatTensor:
        super()._check_dim(samples)
        return self.c1.constraint(samples) + self.c2.constraint(samples)
    
    def gradient(self, samples: torch.FloatTensor, func = None) -> torch.FloatTensor:
        return self.c1.gradient(samples, func) + self.c2.gradient(samples, func)

    def plot(self, fig, ax):
        if type(self.c1) == CircleInequality and type(self.c2) == CircleInequality:
            fig, ax = self.c1.plot(fig, ax, color='red')
            fig, ax = self.c2.plot(fig, ax, color='orange')
            return fig, ax
        else:
            raise NotImplementedError("Unknown configuration of constraints")
    
        