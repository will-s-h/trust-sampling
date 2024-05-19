import torch
import numpy as np
from dataset.quaternion import ax_from_6v

class ObstacleAvoidance:
    def __init__(self, obstacles, target, number_of_windows = 2,  number_of_overlapping_frames = 2,contact=True, device='cuda'):
        self.number_of_windows = number_of_windows
        self.number_of_overlapping_frames = number_of_overlapping_frames
        self.device = device

        self.x_offset_normalized = -0.107
        self.y_offset_normalized = -0.1545
        self.contact = contact

        self.x_target = target[0].to(device)
        self.y_target = target[1].to(device)

        self.z_target = 0.85

        self.obstacles = obstacles


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

        # Continuity constraint (don't care about root position; we will just shift it)
        if samples.shape[0] < 3:
            end_slices = torch.cat((samples[:-1, -self.number_of_overlapping_frames:, :4], samples[:-1, -self.number_of_overlapping_frames:, 6:]), dim=2)
            start_slices = torch.cat((samples[1:, :self.number_of_overlapping_frames, :4], samples[1:, :self.number_of_overlapping_frames, 6:]), dim=2)
        else:
            end_slices = torch.cat((samples[:-1, -self.number_of_overlapping_frames:, :4],
                                    samples[:, -self.number_of_overlapping_frames:, 6:]), dim=2)
            start_slices = torch.cat((samples[1:, :self.number_of_overlapping_frames, :4],
                                      samples[:, :self.number_of_overlapping_frames, 6:]), dim=2)

        # if self.contact:
        #     end_slices[..., 4] = end_slices[..., 4] - end_slices[:, 0, 4].unsqueeze(-1) + self.x_offset_normalized
        #     end_slices[..., 5] = end_slices[..., 5] - end_slices[:, 0, 5].unsqueeze(-1) + self.y_offset_normalized
        # else:
        #     raise NotImplementedError("Non-contact case not implemented yet")
        continuity_loss = torch.max(torch.nn.functional.mse_loss(start_slices, end_slices))

        # Begin- and end-point loss
        stacked_samples = self.stack_samples(samples)
        unnorm_samples = self.normalizer.unnormalize(stacked_samples)
        x_start = unnorm_samples[:, 0, 4]
        y_start = unnorm_samples[:, 0, 5]
        x_end = unnorm_samples[:, -1, 4]
        y_end = unnorm_samples[:, -1, 5]
        z_start = unnorm_samples[:, 0, 6]
        z_end = unnorm_samples[:, -1, 6]
        begin_end_loss = (torch.square(x_start) + torch.square(y_start) + torch.square(
            x_end - self.x_target) + torch.square(y_end - self.y_target) + torch.square(
            z_start - self.z_target) + torch.square(z_end - self.z_target))*0.25

        # Obstacle loss
        poses = self.samples_to_poses(stacked_samples) # (1, T, 24, 3)
        obstacle_violations = []
        for obstacle in self.obstacles:
            squared_distance_obstacle_center = torch.square(poses[...,0] - obstacle[0]) + torch.square(poses[...,1] - obstacle[1]) + torch.square(poses[...,2] - obstacle[2])
            obstacle_violation = torch.relu(obstacle[3] - torch.sqrt(squared_distance_obstacle_center))
            obstacle_violations.append(obstacle_violation)
        obstacle_loss = torch.max(torch.stack(obstacle_violations))


        return (continuity_loss, begin_end_loss, obstacle_loss)

    def gradient(self, samples, func=None):
        with torch.enable_grad():
            samples.requires_grad_(True)
            clean_samples = func(samples)[1]
            constraint_losses = self.constraint(clean_samples)

            loss = constraint_losses[0] + 0.05*constraint_losses[1] + 0.1*constraint_losses[2]

            grad = -torch.autograd.grad(torch.mean(loss) , samples)[0]
            return grad


    def stack_samples(self, samples):
        time_length = samples.shape[1]
        long_sample = []
        x_offset = torch.zeros((1,), device=self.device)
        y_offset = torch.zeros((1,), device=self.device)
        for i in range(samples.shape[0] - 1):
            offset = torch.zeros((time_length - self.number_of_overlapping_frames, 139), device=self.device)
            offset[..., 4] = x_offset
            offset[..., 5] = y_offset
            long_sample.append(samples[i, :-self.number_of_overlapping_frames, :] + offset)
            x_offset = x_offset + samples[i, -self.number_of_overlapping_frames, 4] - samples[i, 0, 4]
            y_offset = y_offset + samples[i, -self.number_of_overlapping_frames, 5] - samples[i, 0, 5]
        offset = torch.zeros((time_length, 139), device=self.device)
        offset[..., 4] = x_offset
        offset[..., 5] = y_offset
        long_sample.append(samples[-1] + offset)
        return torch.cat(long_sample, dim=0).unsqueeze(0)