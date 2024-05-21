import torch
import numpy as np


class TrajectoryConstraint3D:
    def __init__(self, traj, controlled_joint_indices, contact=True, device='cuda'):
        self.nr_controlled_joints = len(controlled_joint_indices)
        assert traj.shape[2] == 3 * self.nr_controlled_joints
        self.traj = traj  # should be of shape [batch, 60, n, 3]
        self.root_slice = slice(4, 7) if contact else slice(0, 3)
        self.shape = (traj.shape[0], 60, 139) if contact else (traj.shape[0], 60, 135)
        self.traj = self.traj.to(device)
        self.device = device

    def set_name(self, name):
        self.name = name

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def __str__(self):
        if self.name is None:
            return "TrajectoryConstraint3D"
        return self.name

    def constraint(self, samples):
        # if samples.dim() == 2:
        #     return torch.sum((samples[..., self.root_slice] - self.traj) ** 2)
        # return torch.mean(
        #     torch.mean((samples[..., self.root_slice] - self.traj.repeat(samples.shape[0], 1, 1)) ** 2, dim=-1), dim=-1)
        loss = torch.sqrt(torch.mean(torch.mean(torch.square(samples[..., self.root_slice] - self.traj), dim=-1),
                    dim=-1))
        return loss
    
    def constraint_oneloss(self, samples):
        traj = self.traj
        traj.requires_grad_(True)
        loss = torch.nn.functional.mse_loss(samples[..., self.root_slice], traj)
        loss_per_batch = torch.mean(torch.mean(torch.square(samples[..., self.root_slice] - traj), dim=-1), dim=-1)
        self.normalizing_factor = (loss_per_batch.unsqueeze(-1).unsqueeze(-1) / loss).detach()
        return loss
    
    def batch_normalize_gradient(self, grad):
        assert self.normalizing_factor is not None
        grad *= self.normalizing_factor
        self.normalizing_factor = None
        return grad

    def gradient(self, samples, func=None):
        # func should be of the form lambda x: self.model_predictions(x, cond, time_cond, clip_x_start=self.clip_denoised)
        # sampels should be of shape [n, 60, 139]
        assert samples.dim() == 3
        assert func is not None
        traj = self.traj
        with torch.enable_grad():
            traj.requires_grad_(True)
            samples.requires_grad_(True)
            next_sample = func(samples)[1]
            loss = -torch.nn.functional.mse_loss(next_sample[..., self.root_slice], traj)

            loss_per_batch = -torch.mean(torch.mean(torch.square(next_sample[..., self.root_slice] - traj), dim=-1),
                                         dim=-1)  # have a loss for each sample in the batch

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
        norm_traj = self.normalizer.normalize(dummy_data)[..., self.root_slice].squeeze()  # should be [60, 139]
        xs = norm_traj[:, 0].cpu().numpy()
        ys = norm_traj[:, 1].cpu().numpy()
        zs = np.repeat(self.Z_MEAN, 60)
        ax.plot(xs, ys, zs, zorder=12, linewidth=1.5, alpha=0.5, color='green')