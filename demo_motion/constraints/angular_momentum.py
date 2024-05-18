import torch
import numpy as np
from dataset.quaternion import ax_from_6v

class AngularMomentumConstraint:
    def __init__(self, angular_momentum, target, number_of_windows = 2,  number_of_overlapping_frames = 2,contact=True, device='cuda'):
        self.number_of_windows = number_of_windows
        self.number_of_overlapping_frames = number_of_overlapping_frames
        self.device = device

        self.x_offset_normalized = -0.107
        self.y_offset_normalized = -0.1545
        self.contact = contact

        self.x_target = target[0].to(device)
        self.y_target = target[1].to(device)

        self.angular_momentum = torch.tensor(angular_momentum).to(device)


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
        begin_end_loss = (torch.square(x_start) + torch.square(y_start) + torch.square(
            x_end - self.x_target) + torch.square(y_end - self.y_target))*0.25

        # Angular momentum loss
        poses = self.samples_to_poses(stacked_samples)
        poses_xzplane = torch.stack((poses[...,[10,11,22,23], 0], poses[...,[10,11,22,23], 2]), dim=3)
        root_relative_poses_xzplane = poses_xzplane - poses_xzplane[..., 0:1, :]

        joint_velocities = poses[:, 1:, ...] - poses[:, :-1, ...]
        joint_velocities_xzplane = torch.stack((joint_velocities[...,[10,11,22,23], 0], joint_velocities[...,[10,11,22,23], 2]), dim=3)
        root_relative_joint_velocities_xzplane = joint_velocities_xzplane - joint_velocities_xzplane[..., 0:1, :]

        cross_input_1 = root_relative_joint_velocities_xzplane
        cross_input_2 = root_relative_poses_xzplane[:, :-1, ...]

        # pad last dimension that is 3 dimensional
        cross_input_1 = torch.cat((cross_input_1, torch.zeros(cross_input_1.shape[:-1] + (1,), device=self.device)), dim=-1)
        cross_input_2 = torch.cat((cross_input_2, torch.zeros(cross_input_2.shape[:-1] + (1,), device=self.device)), dim=-1)
        angular_momentum = torch.sum(torch.cross(cross_input_1, cross_input_2, dim=-1)[...,-1], dim=2)
        angular_momentum = angular_momentum[:, 60:90]
        angular_momentum_violations = torch.relu(self.angular_momentum - angular_momentum)

        angular_momentum_loss = torch.mean(torch.square(angular_momentum_violations))


        return (continuity_loss, begin_end_loss, angular_momentum_loss)

    def gradient(self, samples, func=None):
        with torch.enable_grad():
            samples.requires_grad_(True)
            clean_samples = func(samples)[1]
            constraint_losses = self.constraint(clean_samples)

            loss = constraint_losses[0] + 0*0.05*constraint_losses[1] + constraint_losses[2]

            grad = -torch.autograd.grad(loss , samples)[0]
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