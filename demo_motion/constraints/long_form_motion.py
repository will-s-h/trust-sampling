import torch
import numpy as np


class LongFormMotion:
    def __init__(self, number_of_windows, number_of_overlapping_frames = 5,contact=True, device='cuda'):
        self.number_of_windows = number_of_windows
        self.number_of_overlapping_frames = number_of_overlapping_frames
        self.device = device

        self.x_offset_normalized = -0.107
        self.y_offset_normalized = -0.1545
        self.contact = contact

        self.additional_constraint_1 = False #"covered_distance"
        self.additional_constraint_2 = False # stay in square

    def set_name(self, name):
        self.name = name

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def __str__(self):
        if self.name is None:
            return "LongFormMotion"
        return self.name

    def constraint(self, samples):
        # if samples.dim() == 2:
        #     return torch.sum((samples[..., self.root_slice] - self.traj) ** 2)
        # return torch.mean(
        #     torch.mean((samples[..., self.root_slice] - self.traj.repeat(samples.shape[0], 1, 1)) ** 2, dim=-1), dim=-1)
        loss = torch.sqrt(torch.mean(torch.mean(torch.square(samples[..., self.root_slice] - self.traj), dim=-1),
                    dim=-1))
        return loss

    def gradient(self, samples, func=None):
        # func should be of the form lambda x: self.model_predictions(x, cond, time_cond, clip_x_start=self.clip_denoised)
        # sampels should be of shape [n, 60, 139]
        assert samples.dim() == 3
        assert func is not None
        with torch.enable_grad():
            samples.requires_grad_(True)
            clean_samples = func(samples)[1]

            end_slices = clean_samples[:-1, -self.number_of_windows:, :]
            start_slices = clean_samples[1:, :self.number_of_windows, :]

            if self.contact:
                end_slices[..., 4] = end_slices[..., 4] - end_slices[:, 0, 4].unsqueeze(-1) + self.x_offset_normalized
                end_slices[..., 5] = end_slices[..., 5] - end_slices[:, 0, 5].unsqueeze(-1) + self.y_offset_normalized
            else:
                raise NotImplementedError("Non-contact case not implemented yet")



            loss = -torch.nn.functional.mse_loss(start_slices, end_slices)
            loss_per_batch = -torch.mean(torch.mean(torch.square(start_slices - end_slices), dim=-1),
                                         dim=-1)  # have a loss for each sample in the batch
            # add zero for first window
            loss_per_batch = torch.cat((torch.zeros((1,), device=self.device), loss_per_batch))

            loss_additional_constraint_1 = 0
            if self.additional_constraint_1:
                # move at 3m/s
                distance_covered = torch.tensor((self.number_of_windows * 60 - (self.number_of_windows * self.number_of_overlapping_frames))/20*1, device=self.device)
                distance = []
                unnorm_samples = self.normalizer.unnormalize(clean_samples)
                for i in range(clean_samples.shape[0]-1):
                    x_distance = unnorm_samples[i, 1:-self.number_of_windows+1, 4] - unnorm_samples[i, :-self.number_of_windows, 4]
                    y_distance = unnorm_samples[i, 1:-self.number_of_windows+1, 5] - unnorm_samples[i, :-self.number_of_windows, 5]
                    distance.append(torch.sqrt(x_distance**2 + y_distance**2))
                x_distance = unnorm_samples[i, 1:, 4] - unnorm_samples[i,:-1, 4]
                y_distance = unnorm_samples[i, 1:, 5] - unnorm_samples[i,:-1, 5]
                distance.append(torch.sqrt(x_distance**2 + y_distance**2))
                distance = torch.cat(distance)
                torch.sum(distance)
                loss_additional_constraint_1 = -torch.nn.functional.mse_loss(torch.sum(distance), distance_covered)
                print(loss_additional_constraint_1)
            loss_additional_constraint_2 = 0
            if self.additional_constraint_2:
                stacked_samples = self.stack_samples(clean_samples)
                unnorm_samples = self.normalizer.unnormalize(stacked_samples)
                x = unnorm_samples[..., 4].squeeze()
                y = unnorm_samples[..., 5].squeeze()
                # x_violation = torch.sqrt(torch.relu(torch.cat((torch.zeros((1,), device=self.device),torch.square(x.flatten()) - 1))))
                # y_violation = torch.sqrt(torch.relu(torch.cat((torch.zeros((1,), device=self.device),torch.square(y.flatten()) - 1))))
                loss_additional_constraint_2 = - torch.square(x[-1]) - torch.square(y[-1])
                print(loss_additional_constraint_2)

            return (torch.autograd.grad(loss + loss_additional_constraint_1 + 10*loss_additional_constraint_2, samples)[0] / loss * loss_per_batch.unsqueeze(-1).unsqueeze(-1)).detach()
            # grad = []
            # for j in range(loss.shape[0]):
            #     grad.append(torch.autograd.grad(loss[j], samples, retain_graph=True)[0][j,...])
            # return torch.stack(grad)

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer
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