import torch

class SpecifiedPointConstraint:
    def __init__(self, points, contact=True, device='cuda'):
        self.X_START, self.Y_START = -0.107, -0.1545
        assert type(points) == list
        self.points = points # should be a list of points, [t (0-59), body part (0-138), val (float)]
        for i in range(len(self.points)):
            points[i] = tuple(torch.tensor(points[i][j]).to(device) for j in range(len(points[i])))
        self.shape = (60, 139) if contact else (60, 135)
        
    def set_name(self, name):
        self.name = name

    def __str__(self):
        if self.name is None:
            return "SpecifiedPoint"
        return self.name
    
    def constraint(self, samples):
        loss = 0
        for point in self.points:
            vals = point[2].repeat(samples.shape[0])
            loss += torch.nn.functional.mse_loss(samples[:, point[0], point[1]], vals)
        return loss
        
    def gradient(self, samples, func=None):
        # func should be of the form lambda x: self.model_predictions(x, cond, time_cond, clip_x_start=self.clip_denoised)
        # sampels should be of shape [n, 60, 139]
        assert samples.dim() == 3 and samples.shape[1:] == self.shape
        assert func is not None
        
        with torch.enable_grad():
            samples.requires_grad_(True)
            next_sample = func(samples)[1]
            loss = 0
            for point in self.points:
                vals = point[2].repeat(samples.shape[0])
                loss -= torch.nn.functional.mse_loss(next_sample[:, point[0], point[1]], vals)
            return torch.autograd.grad(loss, samples)[0]