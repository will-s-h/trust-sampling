import torch

class Constraint:
    def __init__(self):
        self.dimension = 2  # the value of m, whenver m is mentioned
    
    def _check_dim(self, samples: torch.FloatTensor):
        if samples.dim() == 1 and samples.size(dim=0) == self.dimension: return
        if samples.dim() == 2 and samples.size(dim=1) == self.dimension: return
        raise ValueError(f"samples must have dimension [{self.dimension}] or [n, {self.dimension}]; instead it has dimension {samples.shape}")
    
    def constraint(self, samples: torch.FloatTensor):
        '''
        samples: [n, m] or [m] tensor
        ---
        returns [n] or [] tensor, the value of the constraint function at each point (should be 0 if constraint is met)
        '''
        raise NotImplementedError(f"constraint() was not defined for this subclass of Constraint.")
    
    def jacobian(self, samples: torch.FloatTensor):
        '''
        sample: [n, m] or [m] tensor
        ---
        returns [1, m] or [n, 1, m] tensor, partial derivatives of the constraint with respect to the first and second variable
        '''
        raise NotImplementedError(f"jacobian() was not defined for this subclass of Constraint.")
    
    def gradient(self, samples: torch.FloatTensor, func = None):
        '''
        in contrast with jacobian, this returns a gradient function that points towards a 0 constraint value.
        
        UPDATE:
        func is now used only for compatibility with constraints of other forms (i.e. for motion constraints)
        -torch.sign is now the default function that the jacobian is multiplied by in order to achieve the gradient
        
        sample: [n, m] or [m] tensor
        ---
        returns [n, m] or [m] tensor
        '''
        raise NotImplementedError(f"gradient() was not defined for this subclass of Constraint.")
    
    def gety(self, x: torch.FloatTensor):
        '''
        x: [n] tensor
        ---
        returns [n] tensor, the y value for each x
        '''
        raise NotImplementedError(f"gety() was not defined for this subclass of Constraint.")
    
    def projection(self, samples: torch.FloatTensor):
        '''
        samples: [n, m] tensor
        ---
        returns [n, m] tensor, where each sample [1, m] is projected onto the constraint
        '''
        raise NotImplementedError(f"projection() was not defined for this subclass of Constraint.")
    
    def kkt(self, samples: torch.FloatTensor):
        '''
        sample: [n, m] or [m] tensor
        ---
        returns [n, m+1, m+1] or [m+1, m+1] tensor, the KKT matrix
        '''
        raise NotImplementedError(f"kkt() was not defined for this subclass of Constraint.")