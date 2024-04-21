import torch
from matplotlib.patches import Circle
from .constraint import Constraint

class CircleInequality(Constraint):
    '''
    Constraint where anything within distance r of x and y is valid
    '''
    
    def __init__(self, x, y, r, slope):
        super().__init__()
        self.x = x
        self.y = y
        self.rsq = r ** 2
        self.slope = slope
        self.relu = torch.nn.ReLU()
    
    def __str__(self):
        return f'circle({self.x},{self.y},{self.rsq},{self.slope})'
    
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
    
    def plot(self, fig, ax, color='red'):
        circle = Circle((self.x, self.y), self.rsq ** 0.5, fill=True, color=color, alpha=0.5)
        ax.add_patch(circle)
        return fig, ax
        