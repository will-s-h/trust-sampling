import torch
from .constraint import Constraint

class LineConstraint(Constraint):
    '''
    Constraint where ax + by = c.
    Alternatively, function f(x,y) = ax + by - c
    '''
    
    def __init__(self, a, b, c):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        
        # initialize variables for projection
        v = torch.Tensor([b, -a])
        v /= torch.norm(v)
        self.proj = torch.outer(v, v)
        self.p_0 = torch.Tensor([0, c/b]) if b != 0 else torch.Tensor([c/a, 0])
    
    def __str__(self):
        return f'line({self.a},{self.b},{self.c})'
    
    def constraint(self, samples: torch.FloatTensor) -> torch.FloatTensor:
        super()._check_dim(samples)
        return self.a * samples[..., 0] + self.b * samples[..., 1] - self.c
    
    def gety(self, x: torch.FloatTensor) -> torch.FloatTensor:
        if self.b == 0: raise ValueError("Cannot gety for vertical line constraint.")
        return (self.c - self.a * x) / self.b
    
    def jacobian(self, samples: torch.FloatTensor) -> torch.FloatTensor:
        if samples is None: samples = torch.zeros(self.dimension)
        super()._check_dim(samples)
        if samples.dim() == 1: return torch.Tensor([[self.a, self.b]])
        else: return torch.Tensor([[[self.a, self.b]]]).expand(samples.shape[0], 1, self.dimension)
    
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
        samples = (samples - self.p_0) @ self.proj.T
        return samples + self.p_0
    
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