import torch
from dataset.quaternion import ax_from_6v

smpl_joints = [
    "root",  # 0
    "lhip",  # 1
    "rhip",  # 2
    "belly", # 3
    "lknee", # 4
    "rknee", # 5
    "spine", # 6
    "lankle",# 7
    "rankle",# 8
    "chest", # 9
    "ltoes", # 10
    "rtoes", # 11
    "neck",  # 12
    "linshoulder", # 13
    "rinshoulder", # 14
    "head", # 15
    "lshoulder", # 16
    "rshoulder",  # 17
    "lelbow", # 18
    "relbow",  # 19
    "lwrist", # 20
    "rwrist", # 21
    "lhand", # 22
    "rhand", # 23
]


class EndEffectorConstraintFootHand:
    def __init__(self, samples, contact=True, device='cuda'):
        # assert type(points) == list

        self.device = device
        self.samples = samples  # should be a list of points, [t, joint, x, y, z]
        self.joint_index = {smpl_joints[i]: i for i in range(len(smpl_joints))}
        self.shape = (60, 139) if contact else (60, 135)

    def str_to_joint(self, string):
        return self.joint_index[string]

    def set_name(self, name):
        self.name = name

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def set_smpl(self, smpl):
        self.smpl = smpl

    def __str__(self):
        if self.name is None:
            return "EndEffector"
        return self.name

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
    def set_targets(self):
        poses = self.samples_to_poses(self.samples)
        left_foot_pos = poses[:, :, 10, :]
        right_hand_pos = poses[:, :, 23, :]
        self.targets = torch.cat((left_foot_pos.unsqueeze(-2), right_hand_pos.unsqueeze(-2)), dim=2)


    def constraint(self, samples):
        loss = 0
        poses = self.samples_to_poses(samples)
        left_foot_pos = poses[:, :, 10, :]
        right_hand_pos = poses[:, :, 23, :]
        generated = torch.cat((left_foot_pos.unsqueeze(-2), right_hand_pos.unsqueeze(-2)), dim=2)
        loss = torch.mean(torch.mean(torch.mean(torch.square(generated - self.targets), dim=-1),
                                     dim=-1), dim=-1)
        return loss
    
    def constraint_oneloss(self, samples):
        loss_per_batch = self.constraint(samples)
        loss = torch.mean(loss_per_batch)
        return loss * (loss_per_batch.unsqueeze(-1).unsqueeze(-1) / loss).detach()

    def constraint_metric(self, samples):
        loss = torch.zeros((samples.shape[0],), device=self.device)
        poses = self.samples_to_poses(samples)
        for point in self.points:
            for dim in range(3):  # x, y, and z
                vals = point[dim + 2].repeat(poses.shape[0])
                loss += torch.square(poses[:, point[0], point[1], dim] - vals)
        return loss / len(self.points) / 3

    def gradient(self, samples, func=None):
        # func should be of the form lambda x: self.model_predictions(x, cond, time_cond, clip_x_start=self.clip_denoised)
        # sampels should be of shape [n, 60, 139]
        assert samples.dim() == 3 and samples.shape[1:] == self.shape
        assert func is not None

        with torch.enable_grad():
            samples.requires_grad_(True)
            next_sample = func(samples)[1]
            loss_per_batch = -self.constraint(next_sample)
            loss = torch.mean(loss_per_batch)
            print('loss', loss)
            return (torch.autograd.grad(loss, samples)[0] / loss * loss_per_batch.unsqueeze(-1).unsqueeze(-1)).detach()
            # return torch.autograd.grad(loss, samples)[0]

    def plot(self, fig, ax):
        xs = [point[2].item() for point in self.points]
        ys = [point[3].item() for point in self.points]
        zs = [point[4].item() for point in self.points]
        ax.scatter(xs, ys, zs, zorder=12, s=10, alpha=0.5, color='green')


class EndEffectorConstraint:
    def __init__(self, points, contact=True, device='cuda'):
        assert type(points) == list
        
        self.device = device
        self.points = points # should be a list of points, [t, joint, x, y, z]
        self.joint_index = {smpl_joints[i]: i for i in range(len(smpl_joints))}
        self.shape = (60, 139) if contact else (60, 135)
        
        # joint could be either string or int
        for i in range(len(self.points)):
            new_tuple = (points[i][0], self.str_to_joint(points[i][1]) if type(points[i][1]) == str else points[i][1], 
                         points[i][2], points[i][3], points[i][4])
            points[i] = tuple(torch.tensor(new_tuple[j]).to(device) for j in range(len(new_tuple)))
    
    def str_to_joint(self, string):
        return self.joint_index[string]
        
    def set_name(self, name):
        self.name = name
        
    def set_normalizer(self, normalizer):
        self.normalizer = normalizer
    
    def set_smpl(self, smpl):
        self.smpl = smpl

    def __str__(self):
        if self.name is None:
            return "EndEffector"
        return self.name
    
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
        loss = 0
        poses = self.samples_to_poses(samples)
        for point in self.points:
            for dim in range(3): # x, y, and z
                vals = point[dim+2].repeat(poses.shape[0])
                loss += torch.nn.functional.mse_loss(poses[:, point[0], point[1], dim], vals)
        return loss

    def constraint_metric(self, samples):
        loss = torch.zeros((samples.shape[0],), device=self.device)
        poses = self.samples_to_poses(samples)
        for point in self.points:
            for dim in range(3): # x, y, and z
                vals = point[dim+2].repeat(poses.shape[0])
                loss += torch.square(poses[:, point[0], point[1], dim]- vals)
        return loss / len(self.points) / 3
        
    def gradient(self, samples, func=None):
        # func should be of the form lambda x: self.model_predictions(x, cond, time_cond, clip_x_start=self.clip_denoised)
        # sampels should be of shape [n, 60, 139]
        assert samples.dim() == 3 and samples.shape[1:] == self.shape
        assert func is not None
        
        with torch.enable_grad():
            samples.requires_grad_(True)
            next_sample = func(samples)[1]
            loss = -self.constraint(next_sample)
            return torch.autograd.grad(loss, samples)[0]
    
    def plot(self, fig, ax):
        xs = [point[2].item() for point in self.points]
        ys = [point[3].item() for point in self.points]
        zs = [point[4].item() for point in self.points]
        ax.scatter(xs, ys, zs, zorder=12, s=10, alpha=0.5, color='green')