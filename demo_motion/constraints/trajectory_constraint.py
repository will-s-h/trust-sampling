import torch
import numpy as np

class TrajectoryConstraint:
    def __init__(self, traj, contact=True, device='cuda'):
        self.X_START, self.Y_START = -0.107, -0.1545
        self.Z_MEAN = 0.9719441533088684
        self.traj = traj # should be of shape [60, 2]
        self.root_slice = slice(4, 6) if contact else slice(0, 2)
        self.shape = (60, 139) if contact else (60, 135)
        self.traj = self.traj.to(device)
        self.device = device
        
        assert self.traj.shape == (self.shape[0], 2)
    
    def set_name(self, name):
        self.name = name
    
    def set_normalizer(self, normalizer):
        self.normalizer = normalizer
    
    def __str__(self):
        if self.name is None:
            return "TrajectoryConstraint"
        return self.name

    def naive_gradient(self, samples): # should be of shape [n, 60, 139]
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
    
    def constraint(self, samples):
        if samples.dim() == 2:
            return torch.sum((samples[..., self.root_slice] - self.traj) ** 2)
        return torch.mean(torch.mean((samples[..., self.root_slice] - self.traj.repeat(samples.shape[0], 1, 1)) ** 2, dim=-1 ), dim=-1)
    
    def constraint_oneloss(self, samples):
        traj = self.traj.repeat(samples.shape[0], 1, 1)
        traj.requires_grad_(True)
        loss = -torch.nn.functional.mse_loss(samples[..., self.root_slice], traj)
        loss_per_batch = -torch.mean(torch.mean(torch.square(samples[..., self.root_slice]- traj), dim=-1),dim=-1) # have a loss for each sample in the batch
        return loss * (loss_per_batch.unsqueeze(-1).unsqueeze(-1) / loss).detach()
        
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

            loss_per_batch = -torch.mean(torch.mean(torch.square(next_sample[..., self.root_slice]- traj), dim=-1),dim=-1) # have a loss for each sample in the batch

            return (torch.autograd.grad(loss, samples)[0] / loss * loss_per_batch.unsqueeze(-1).unsqueeze(-1)).detach()
            # grad = []
            # for j in range(loss.shape[0]):
            #     grad.append(torch.autograd.grad(loss[j], samples, retain_graph=True)[0][j,...])
            # return torch.stack(grad)

    def plot(self, fig, ax):
        # first, normalize traj
        assert self.normalizer is not None, "must have initialized a normalizer via set_normalizer()"
        dummy_data = torch.zeros((1,) + self.shape, device=self.device)
        dummy_data[..., self.root_slice] = self.traj
        norm_traj = self.normalizer.normalize(dummy_data)[..., self.root_slice].squeeze() # should be [60, 139]
        xs = norm_traj[:, 0].cpu().numpy()
        ys = norm_traj[:, 1].cpu().numpy()
        zs = np.repeat(self.Z_MEAN, 60)
        ax.plot(xs, ys, zs, zorder=12, linewidth=1.5, alpha=0.5, color='green')