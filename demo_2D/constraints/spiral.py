import torch
import numpy as np
from .constraint import Constraint

class SpiralConstraint(Constraint):
    # defines r = theta / theta_0 from 0 <= theta <= max_theta
    def __init__(self, theta_0=3*np.pi, max_theta=6*np.pi):
        super().__init__()
        self.theta_0 = theta_0
        self.max_theta = max_theta
    
    def constraint(self, sample: tuple | torch.FloatTensor) -> torch.FloatTensor:
        if type(sample) == torch.Tensor and (sample.dim() != 1 or sample.size(dim=0) != self.dimension):
            raise ValueError(f"sample must be one-dimensional, with dimension length {self.dimension}")
        elif type(sample) == tuple and len(sample) != 2:
            raise ValueError(f"sample must be of size {self.dimension}, instead it is of size {len(sample)}")
        
        r = (sample[0]**2 + sample[1]**2)**0.5
        theta = np.arctan2(sample[1], sample[0])
        min_constraint = abs(r - theta/(3*np.pi))
        while theta <= self.max_theta + 2*np.pi:
            min_constraint = min(min_constraint, abs(r - theta/(3*np.pi)))
            theta += 2*np.pi
        return min_constraint