import torch

class TrajectoryConstraint:
    def __init__(self, traj, contact=True, device='cuda'):
        self.X_START, self.Y_START = -0.107, -0.1545
        self.traj = traj # should be of shape [60, 2]
        self.root_slice = slice(4, 6) if contact else slice(0, 2)
        self.shape = (60, 139) if contact else (60, 135)
        self.traj = self.traj.to(device)
        
        assert self.traj.shape == (self.shape[0], 2)

    def traj_constraint(self, samples): # should be of shape [n, 60, 139]
        # if samples is passed in to have trajectory 0, this means that we wish to 
        if torch.equal(samples[..., self.root_slice], torch.zeros_like(samples[..., self.root_slice])):
            samples[..., self.root_slice.start] = self.X_START
            samples[..., self.root_slice.start+1] = self.Y_START
        
        if samples.dim() == 2: # of shape [60, 139]
            assert samples.shape == self.shape
            grad = torch.zeros_like(samples)
            grad[..., self.root_slice] = self.traj - samples[..., self.root_slice]
            return grad
        else:
            # otherwise, of shape [n, 60, 139]
            assert samples.shape[1:] == self.shape
            grad = torch.zeros_like(samples) 
            grad[..., self.root_slice] = self.traj.repeat(grad.shape[0], 1, 1) - samples[..., self.root_slice] 
            return grad
        
    def gradient(self, samples, func=None):
        # func should be of the form lambda x: self.model_predictions(x, cond, time_cond, clip_x_start=self.clip_denoised)
        # sampels should be of shape [n, 60, 139]
        assert samples.dim() == 3 and samples.shape[1:] == self.shape
        assert func is not None
        traj = self.traj.repeat(samples.shape[0], 1, 1)
        with torch.enable_grad():
            traj.requires_grad_(True)
            samples.requires_grad_(True)
            next_sample = func(samples)[1]
            loss = -torch.nn.functional.mse_loss(next_sample[..., self.root_slice], traj)
            return (torch.autograd.grad(loss, samples)[0])