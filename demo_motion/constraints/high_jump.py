import torch
import numpy as np
from dataset.quaternion import ax_from_6v

class HighJumpConstraint:
    def __init__(self, jump_height, target, number_of_windows = 2,  number_of_overlapping_frames = 2,contact=True, device='cuda'):
        self.number_of_windows = number_of_windows
        self.number_of_overlapping_frames = number_of_overlapping_frames
        self.device = device

        self.x_offset_normalized = -0.107
        self.y_offset_normalized = -0.1545
        self.contact = contact

        self.x_target = target[0].to(device)
        self.y_target = target[1].to(device)

        self.z_target = 0.85

        self.jump_height = torch.tensor(jump_height).to(device)

    def set_name(self, name):
        self.name = name

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def __str__(self):
        if self.name is None:
            return "Obstacle"
        return self.name

    def set_smpl(self, smpl):
        self.smpl = smpl

    def samples_to_poses(self, samples):
        # normalizer and smpl object must be set
        assert self.normalizer is not None, "must set normalizer via set_normalizer()"
        assert self.smpl is not None, "must set smpl object via set_smpl()"

        samples = self.normalizer.unnormalize(samples)

        # discard contact
        if samples.shape[2] != 135:
            _, samples = torch.split(
                samples, (4, samples.shape[2] - 4), dim=2
            )

        # do the FK all at once
        b, s, c = samples.shape
        pos = samples[:, :, :3].to('cuda')
        q = samples[:, :, 3:].reshape(b, s, 22, 6)
        # go 6d to ax
        q = ax_from_6v(q).to('cuda')
        poses = self.smpl.forward(q, pos)
        return poses


    def constraint(self, samples):
        # Continuity constraint - NA in this case, we generate a 3s motion
        continuity_loss = torch.zeros((1,), device=self.device)

        # Begin- and end-point loss
        unnorm_samples = self.normalizer.unnormalize(samples)
        x_start = unnorm_samples[:, 0, 4]
        y_start = unnorm_samples[:, 0, 5]
        x_end = unnorm_samples[:, -1, 4]
        y_end = unnorm_samples[:, -1, 5]
        z_start = unnorm_samples[:, 0, 6]
        z_end = unnorm_samples[:, -1, 6]
        begin_end_loss = (torch.square(x_start) + torch.square(y_start) + torch.square(
            x_end - self.x_target) + torch.square(y_end - self.y_target) + torch.square(
            z_start - self.z_target) + torch.square(z_end - self.z_target))*0.25

        # High hand loss
        poses = self.samples_to_poses(samples) # (1, T, 24, 3)
        vertical_location = poses[:,30,:,2]
        high_hand_loss = torch.max(torch.square(torch.relu(self.jump_height - vertical_location)), dim=1)[0]

        return (continuity_loss, begin_end_loss, high_hand_loss)

    def vertical_location_violation(self, samples):
        poses = self.samples_to_poses(samples)
        vertical_location = poses[:,30,:,2]
        violation = torch.max(self.jump_height - vertical_location, dim=1)[0]
        # replace negative numbers by zero, exactly, not relu
        violation = torch.where(violation < 0, torch.zeros_like(violation), violation)
        return violation

    def constraint_oneloss(self, samples):
        continuity_loss, begin_end_loss, high_hand_loss = self.constraint(samples)
        loss = continuity_loss + begin_end_loss + high_hand_loss
        self.normalizing_factor = (loss.unsqueeze(-1).unsqueeze(-1) / torch.mean(loss)).detach() 
        return torch.mean(loss)
    
    def batch_normalize_gradient(self, grad):
        assert self.normalizing_factor is not None
        grad *= self.normalizing_factor
        self.normalizing_factor = None
        return grad
        
    def gradient(self, samples, func=None):
        with torch.enable_grad():
            samples.requires_grad_(True)
            clean_samples = func(samples)[1]
            constraint_losses = self.constraint(clean_samples)

            # loss = constraint_losses[0] + 0.05*constraint_losses[1] + constraint_losses[2]
            loss = constraint_losses[0] + constraint_losses[1] + constraint_losses[2]

            grad = -torch.autograd.grad(torch.mean(loss) , samples)[0] * loss.unsqueeze(-1).unsqueeze(-1) / torch.mean(loss)
            return grad

