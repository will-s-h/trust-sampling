import torch
from .constraint import Constraint

class CubicConstraint(Constraint):
    '''
    Constraint where y = ax^3 + bx^2 + cx + d
    Alternatively, function f(x,y) = y - ax^3 - bx^2 - cx - d = 0
    '''
    
    def __init__(self, a, b, c, d):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        self.d = d
    
    def __str__(self):
        return f'cubic({self.a},{self.b},{self.c},{self.d})'
    
    def constraint(self, samples: torch.FloatTensor) -> torch.FloatTensor:
        super()._check_dim(samples)
        return samples[..., 1] - self.a * samples[..., 0]**3 - self.b * samples[..., 0]**2 \
               - self.c * samples[..., 0] - self.d
    
    def gety(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.a * x**3 + self.b * x **2 + self.c * x + self.d
    
    def jacobian(self, samples: torch.FloatTensor) -> torch.FloatTensor:
        if samples is None: samples = torch.zeros(self.dimension)
        super()._check_dim(samples)
        if samples.dim() == 1: 
            return torch.Tensor([[-3 * self.a * samples[:, 0]**2 - 2 * self.b * samples[:, 0] - self.c, 1]])
        else:
            xderiv = -3 * self.a * samples[:, 0]**2 - 2 * self.b * samples[:, 0] - self.c
            yderiv = torch.ones(len(samples))
            return torch.cat((xderiv.unsqueeze(1), yderiv.unsqueeze(1)), dim=1).unsqueeze(1)
    
    def gradient(self, samples: torch.FloatTensor, func = torch.sign) -> torch.FloatTensor:
        super()._check_dim(samples)
        factor = func(self.constraint(samples))
        if samples.dim() == 1:
            J = self.jacobian(samples).squeeze(0)
            return factor * J
        else:
            J = self.jacobian(samples).squeeze(1)
            return (factor * J.T).T
    
    def projection(self, samples: torch.FloatTensor) -> torch.FloatTensor:
        super()._check_dim(samples)
        samples[..., 1] = self.gety(samples[..., 0])
        return samples
    
    def kkt(self, samples: torch.FloatTensor) -> torch.FloatTensor:
        super()._check_dim(samples)
        J = self.jacobian(samples) # n x 1 x m tensor
        if samples.dim() == 1: J.unsqueeze(0)
        
        n = J.shape[0]
        kkt = torch.zeros((n, self.dimension + 1, self.dimension + 1))
        kkt[:, :self.dimension, :self.dimension] = torch.eye(self.dimension).unsqueeze(0).expand(n, self.dimension, self.dimension) if samples.grad is None \
                                                   else torch.diag_embed(samples.grad)
        kkt[:, self.dimension:, :self.dimension] = J
        kkt[:, :self.dimension, self.dimension:] = torch.swapaxes(J, 1, 2)
        if samples.dim() == 1: kkt = kkt.squeeze(0)
        return kkt