import torch
import numpy as np
from .constraint import Constraint

class PolyLikeConstraint(Constraint):
    '''
    Constraint a x^xexp + b y^yexp = c
    '''
    
    def __init__(self, a, xexp, b, yexp, c):
        super().__init__()
        self.a = a
        self.xexp = xexp
        self.b = b
        self.yexp = yexp
        self.c = c
    
    def __str__(self):
        return f'polylike({self.a}x^{self.xexp}+{self.b}y^{self.yexp}={self.c})'
    
    def constraint(self, samples: torch.FloatTensor) -> torch.FloatTensor:
        super()._check_dim(samples)
        return self.a * samples[..., 0] ** self.xexp + self.b * samples[..., 1] ** self.yexp - self.c
    
    def gradient(self, samples: torch.FloatTensor, func = None) -> torch.FloatTensor:
        deriv = samples.clone()
        deriv[..., 0] = self.a * self.xexp * deriv[..., 0] ** (self.xexp - 1)
        deriv[..., 1] = self.b * self.yexp * deriv[..., 1] ** (self.yexp - 1)
        deriv = (deriv.T * -torch.sign(self.constraint(samples))).T
        return deriv
    
    def _nth_root(x, n):
        return np.sign(x) * np.power(np.abs(x), 1/n)
    
    def plot(self, fig, ax):
        xs = np.linspace(-2, 2, 200)
        ys = PolyLikeConstraint._nth_root((self.c - self.a * xs ** self.xexp)/self.b, self.yexp)
        xs = xs[(-2 <= ys) & (ys <= 2)]
        ys = ys[(-2 <= ys) & (ys <= 2)]
        ax.plot(xs, ys, alpha=0.5)
        return fig, ax
        