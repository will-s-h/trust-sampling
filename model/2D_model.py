from itertools import repeat
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from model.positional_embeddings import PositionalEmbedding


class Block(nn.Module):
    def __init__(self, size: int):
        super().__init__()

        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        return x + self.act(self.ff(x))


class MLP(nn.Module):
    def __init__(self, hidden_size: int = 128, hidden_layers: int = 3, emb_size: int = 128,
                 time_emb: str = "sinusoidal", input_emb: str = "sinusoidal"):
        super().__init__()

        self.time_mlp = PositionalEmbedding(emb_size, time_emb)
        self.input_mlp1 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.input_mlp2 = PositionalEmbedding(emb_size, input_emb, scale=25.0)

        concat_size = len(self.time_mlp.layer) + \
            len(self.input_mlp1.layer) + len(self.input_mlp2.layer)
        layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size))
        layers.append(nn.Linear(hidden_size, 2))
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x, t):
        x1_emb = self.input_mlp1(x[:, 0])
        x2_emb = self.input_mlp2(x[:, 1])
        t_emb = self.time_mlp(t)
        x = torch.cat((x1_emb, x2_emb, t_emb), dim=-1)
        x = self.joint_mlp(x)
        return x

class NoiseScheduler():
    def __init__(self,
                 num_timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02,
                 beta_schedule="linear"):

        self.num_timesteps = num_timesteps
        if beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_timesteps, dtype=torch.float32)
        elif beta_schedule == "quadratic":
            self.betas = torch.linspace(
                beta_start ** 0.5, beta_end ** 0.5, num_timesteps, dtype=torch.float32) ** 2

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.)

        # required for self.add_noise
        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5

        # required for reconstruct_x0
        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(
            1 / self.alphas_cumprod - 1)

        # required for q_posterior
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

    def reconstruct_x0(self, x_t, t, noise):
        s1 = self.sqrt_inv_alphas_cumprod[t]
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        return s1 * x_t - s2 * noise

    def q_posterior(self, x_0, x_t, t):
        s1 = self.posterior_mean_coef1[t]
        s2 = self.posterior_mean_coef2[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        mu = s1 * x_0 + s2 * x_t
        return mu

    def get_variance(self, t):
        if t == 0:
            return 0

        variance = self.betas[t] * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t])
        variance = variance.clip(1e-20)
        return variance

    def step(self, model_output, timestep, sample, add_noise=False):
        t = timestep
        pred_original_sample = self.reconstruct_x0(sample, t, model_output)  
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)
        
        if add_noise and t > 0:
            noise = torch.randn_like(model_output)
            variance = (self.get_variance(t) ** 0.5) * noise
            pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample
    
    def trust_step(self, model_output, timestep, sample, add_noise=False):
        t = timestep
        
        pred_original_sample = self.reconstruct_x0(sample, t, model_output)  
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)
        
        if add_noise and t > 0:
            noise = torch.randn_like(model_output)
            variance = (self.get_variance(t) ** 0.5) * noise
            pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample
    
    def _solve_kkt(kkt_i, b_i, sample_i):
        result = np.linalg.solve(kkt_i, b_i)
        s_p = result[:sample_i.shape[0]]
        return sample_i + s_p
    
    def constrained_step(self, model_output, timestep, sample, constraint_func, kkt_func, add_noise=False):
        t = timestep
        pred_original_sample = self.reconstruct_x0(sample, t, model_output)  
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)
        hessian = pred_prev_sample.requires_grad
        
        if hessian:
            pred_prev_sample.sum().backward()
            pred_prev_sample = pred_prev_sample.detach()
        
        # calculate values for matrix equation
        with torch.no_grad():
            constraint_values = -constraint_func(sample)
            if callable(kkt_func):
                kkts = kkt_func(sample)
            s = pred_prev_sample - sample
            b = np.concatenate((s, constraint_values.unsqueeze(1)), axis=1)
            sample = sample.detach()
        
        # solve KKT equation
        for i, args in enumerate(zip(kkts if callable(kkt_func) else repeat(kkt_func), b, sample)):
            pred_prev_sample[i] = NoiseScheduler._solve_kkt(*args)
        
        # NOT in-manifold noise
        if add_noise and t > 0:
            noise = torch.randn_like(model_output)
            variance = (self.get_variance(t) ** 0.5) * noise
            pred_prev_sample = pred_prev_sample + variance

        assert not pred_prev_sample.requires_grad
        return pred_prev_sample
    
    def guided_step(self, model_output, timestep, sample, constraint_gradient, w=0.1, add_noise=False):
        t = timestep
        
        if sample.requires_grad: # performing MCG correction
            x_0_original = self.reconstruct_x0(sample, t, model_output)
            x_0_original.sum().backward()
            grad = sample.grad
            with torch.no_grad():
                guidance = w * self.sqrt_one_minus_alphas_cumprod[t] * constraint_gradient(x_0_original)
                # project guidance onto gradient
                guidance = (torch.sum(grad * guidance, dim=1) / (torch.linalg.vector_norm(grad, dim=1) * grad.T)).T
        else:
            guidance = w * self.sqrt_one_minus_alphas_cumprod[t] * constraint_gradient(sample)
        
        with torch.no_grad():
            model_output += guidance
            pred_original_sample = self.reconstruct_x0(sample, t, model_output)
            pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)
        
        if add_noise and t > 0:
            noise = torch.randn_like(model_output)
            variance = (self.get_variance(t) ** 0.5) * noise
            pred_prev_sample = pred_prev_sample + variance
            
        return pred_prev_sample
            

    def add_noise(self, x_start, x_noise, timesteps):
        s1 = self.sqrt_alphas_cumprod[timesteps]
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]

        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)

        return s1 * x_start + s2 * x_noise

    def __len__(self):
        return self.num_timesteps