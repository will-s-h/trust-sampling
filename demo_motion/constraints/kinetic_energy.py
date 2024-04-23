import torch
from dataset.quaternion import ax_from_6v

class KineticEnergyConstraint:
    def __init__(self, KE, contact=True, device='cuda'):
        self.device = device
        self.KE = KE
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
            return "KineticEnergy"
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
        energy = torch.sum((poses[:, 1:] - poses[:, :-1]) ** 2)
        loss = torch.nn.functional.mse_loss(energy, torch.tensor(self.KE).float().to(self.device) * samples.shape[0])
        return loss
        
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