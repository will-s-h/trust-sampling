import torch

class Combine():
    '''
    Combine two constraints by adding their gradients
    '''
    
    def __init__(self, c1, c2):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
    
    def __str__(self):
        if self.name is None:
            return f'{self.c1}+{self.c2}'
        return self.name
    
    def set_name(self, name):
        self.name = name
    
    def constraint(self, samples: torch.FloatTensor) -> torch.FloatTensor:
        return self.c1.constraint(samples) + self.c2.constraint(samples)
    
    def gradient(self, samples: torch.FloatTensor, func = None) -> torch.FloatTensor:
        return self.c1.gradient(samples, func) + self.c2.gradient(samples, func)

    def plot(self, fig, ax):
        if hasattr(self.c1, 'plot'):
            self.c1.plot(fig, ax)
        if hasattr(self.c2, 'plot'):
            self.c2.plot(fig, ax)
    
        