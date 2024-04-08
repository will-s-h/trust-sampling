import copy
from functools import partial
import numpy as np
import torch
import random 
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from tqdm import tqdm
from dataset.quaternion import ax_from_6v

from .utils import extract, make_beta_schedule

def identity(t, *args, **kwargs):
    return t

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        horizon,
        repr_dim,
        smpl,
        n_timestep=1000,
        schedule="linear",
        loss_type="l1",
        clip_denoised=True,
        predict_epsilon=True,
        guidance_weight=3,
        use_p2=False,
        cond_drop_prob=0.2,
        predict_contact=False,
    ):
        super().__init__()
        self.model = model  # this model is the either a MotionDecoder or 2D model
        self.ema = EMA(0.9999)
        self.master_model = copy.deepcopy(self.model)
        self.predict_contact = predict_contact

        # make a SMPL instance for FK module
        self.smpl = smpl

        betas = torch.Tensor(
            make_beta_schedule(schedule=schedule, n_timestep=n_timestep)
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timestep = int(n_timestep)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # p2 weighting
        self.p2_loss_weight_k = 1
        self.p2_loss_weight_gamma = 0.5 if use_p2 else 0
        self.register_buffer(
            "p2_loss_weight",
            (self.p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod))
            ** -self.p2_loss_weight_gamma,
        )

        ## get loss coefficients and initialize objective
        self.loss_fn = F.mse_loss if loss_type == "l2" else F.l1_loss

    # ------------------------------------------ sampling ------------------------------------------#
    
    # predict epsilon from x0
    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    # predict epsilon and x0 from x_t
    def model_predictions(self, x, cond, t, clip_x_start = False):
        model_output = self.model.forward(x, cond, t)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity
        
        x_start = model_output
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return pred_noise, x_start

        
    @torch.no_grad()
    def ddim_sample(self, shape, cond, sample_steps=50, **kwargs):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.n_timestep, sample_steps, 1

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn(shape, device = device)
        cond = cond.to(device)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            # produce a bunch of times, for each of the samples in the batch
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            
            # predict completely denoised product
            pred_noise, x_start, *_ = self.model_predictions(x, cond, time_cond, clip_x_start = self.clip_denoised)

            if time_next < 0:
                x = x_start
                continue
            
            # apply diffusion noise again, except with one less step of noise than what was denoised.
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x)

            x = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
                  
            new_pred_noise, *_ = self.model_predictions(x, cond, time_cond, clip_x_start=self.clip_denoised)
            print(f'reached norm: {torch.norm(new_pred_noise).item()}')
        return x
    
    def _get_trajectory(self, x):
        return x[..., 4:6].squeeze(0).cpu() if x.dim() == 3 and x.shape[0] == 1 else x[..., 4:6].cpu()
    
    @torch.no_grad()
    def trust_sample(self, shape, cond, sample_steps=50, constraint_obj=None, debug=False, **kwargs):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.n_timestep, sample_steps, 1
        assert constraint_obj is not None

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        
        x = torch.randn(shape, device = device)
        cond = cond.to(device)

        x_start = None
        traj = [(self._get_trajectory(x), 1001, 'starting trajectory')]

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            # just repeat each step 5 times
            for _ in range(5 if time_next < 0 else 1):
                
                # produce a bunch of times, for each of the samples in the batch
                time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
                
                # predict completely denoised product
                pred_noise, x_start, *_ = self.model_predictions(x, cond, time_cond, clip_x_start = self.clip_denoised)
                traj.append((self._get_trajectory(x_start), time, 'diffusion step'))
                if time_next < 0:
                    pred_noise, *_ = self.model_predictions(x_start, cond, time_cond, clip_x_start = self.clip_denoised)
                    print(f'final norm: {torch.norm(pred_noise).item()}')
                    x = x_start
                    continue
                
                # apply diffusion noise again, except with one less step of noise than what was denoised.
                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]

                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()

                noise = torch.randn_like(x)
                
                #### new stuff
                model_mean = x_start * alpha_next.sqrt() + c * pred_noise
                new_pred_noise, *_ = self.model_predictions(model_mean, cond, time_cond, clip_x_start=self.clip_denoised)
                print(f'reached norm: {torch.norm(new_pred_noise).item()}')
                j = 0
                
                NORM_UPPER_BOUND = 80
                ITERATIONS_MAX   = 5
                while j < ITERATIONS_MAX and torch.norm(new_pred_noise).item() <= NORM_UPPER_BOUND:
                    const = c * extract(self.sqrt_one_minus_alphas_cumprod, time_cond, x.shape)
                    
                    # experiment 1: equivalent to less drastic inpainting
                    # g = (const if time < 200 else 1) * constraint_obj.traj_constraint(model_mean if time < 200 else torch.zeros_like(model_mean)) # gradient_f(x)
                    
                    # experiment 2: with backprop
                    # g = (1) * constraint_obj.traj_constraint(torch.zeros_like(model_mean)) if time > 200 else \
                    #     constraint_obj.traj_constraint_backprop(model_mean, lambda x: self.model_predictions(x, cond, time_cond, clip_x_start=self.clip_denoised))
                    
                    # if time <= 200: 
                    #     g *= 1 / torch.norm(g).item()
                    
                    # experiment 3: only backprop
                    g = (1) * constraint_obj.traj_constraint_backprop(model_mean, lambda x: self.model_predictions(x, cond, time_cond, clip_x_start=self.clip_denoised))
                    g *= 1 / torch.norm(g).item()
                    
                    print(f'size of gradient step: {torch.norm(g).item()}')
                    model_mean = model_mean + g
                    traj.append((self._get_trajectory(model_mean), time, 'gradient step'))
                    j += 1
                    new_pred_noise, *_ = self.model_predictions(model_mean, cond, time_cond, clip_x_start=self.clip_denoised)
                
                x = model_mean + sigma * noise
                new_pred_noise, *_ = self.model_predictions(x, cond, time_cond, clip_x_start=self.clip_denoised)
                traj.append((self._get_trajectory(x), time, 'noise'))
                print(f'{j} iterations, reached norm: {torch.norm(new_pred_noise).item()}')
                
                #### next stuff

                # x = x_start * alpha_next.sqrt() + \
                #     c * pred_noise + \
                #     sigma * noise
        return (x, traj) if debug else x
    
    @torch.no_grad()
    def _get_normalized_loc(self, normalizer, x, y, device='cuda'):
        X_COORD = 4 if self.predict_contact else 0
        frame = torch.zeros((1, 1, 135 + X_COORD))
        frame[0, 0, X_COORD] = x
        frame[0, 0, X_COORD + 1] = y
        return normalizer.normalize(frame)[0, 0, X_COORD : X_COORD + 2].to(device)


    # ------------------------------------------ training ------------------------------------------#
    
    def q_sample(self, x_start, t, noise=None):
        '''
        x_start = [batch_size x 60 x 135]
        t = [batch_size]
        noise = [batch_size x 60 x 135]

        noises x_start by t steps
        '''
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample
    
    def p_losses(self, x_start, cond, t, normalizer=None):
        '''
        x_start : [batch_size x 60 x 135]
        cond : None
        t : [batch_size], randomly generated
        '''
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        mask = None
        if self.model.use_masks:
            mask = self.create_random_mask(x_start)
            x_noisy = mask * x_start + (1 - mask) * x_noisy

        # reconstruct
        x_recon = self.model(x_noisy, mask if self.model.use_masks else cond, t)
        assert noise.shape == x_recon.shape

        model_out = x_recon
        if self.predict_epsilon:
            target = noise
        else:
            target = x_start

        # full reconstruction loss
        loss = self.loss_fn(model_out, target, reduction="none")
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss * extract(self.p2_loss_weight, t, loss.shape)

        # inpaint loss
        inpaint_loss = None
        if self.model.use_masks:
            inpaint_loss = self.loss_fn(model_out * mask, target * mask, reduction="none")
            inpaint_loss = reduce(inpaint_loss, "b ... -> b (...)", "mean")
            inpaint_loss = inpaint_loss * extract(self.p2_loss_weight, t, inpaint_loss.shape)

        # root motion extra loss
        loss_root = self.loss_fn(model_out[...,:3], target[...,:3], reduction="none") if not self.predict_contact else \
                    self.loss_fn(model_out[...,4:7], target[...,4:7], reduction="none")
        loss_root = reduce(loss_root, "b ... -> b (...)", "mean")
        loss_root = loss_root * extract(self.p2_loss_weight, t, loss_root.shape)

        if not self.predict_contact:
            losses = (
                0.636 * loss.mean(),
                1.000 * loss_root.mean(),
                torch.tensor(0.0,device='cuda'),
                torch.tensor(0.0,device='cuda'),
                2.000 * inpaint_loss.mean() if self.model.use_masks else torch.tensor(0.0, device='cuda')
            )
            return sum(losses), losses
        
        # split off contact from the rest
        model_contact, model_out = torch.split(
            model_out, (4, model_out.shape[2] - 4), dim=2
        )
        target_contact, target = torch.split(target, (4, target.shape[2] - 4), dim=2)

        # velocity loss
        # target_v = target[:, 1:] - target[:, :-1]
        # model_out_v = model_out[:, 1:] - model_out[:, :-1]
        # v_loss = self.loss_fn(model_out_v, target_v, reduction="none")
        # v_loss = reduce(v_loss, "b ... -> b (...)", "mean")
        # v_loss = v_loss * extract(self.p2_loss_weight, t, v_loss.shape)
        
        # FK loss
        b, s, c = model_out.shape
        # unnormalize
        assert normalizer is not None, "must provide normalizer if predicting contact"
        model_out = normalizer.unnormalize(model_out)
        target = normalizer.unnormalize(target)
        # X, Q
        model_x = model_out[:, :, :3]
        model_q = ax_from_6v(model_out[:, :, 3:].reshape(b, s, -1, 6))  # should be b, s, 22, 6
        target_x = target[:, :, :3]
        target_q = ax_from_6v(target[:, :, 3:].reshape(b, s, -1, 6))  # should be b, s, 22, 6
        
        # perform FK
        model_xp = self.smpl.forward(model_q, model_x)
        target_xp = self.smpl.forward(target_q, target_x)

        fk_loss = self.loss_fn(model_xp, target_xp, reduction="none")
        fk_loss = reduce(fk_loss, "b ... -> b (...)", "mean")
        fk_loss = fk_loss * extract(self.p2_loss_weight, t, fk_loss.shape)
        
        # foot skate loss
        foot_idx = [7, 8, 10, 11]
        
        # find static indices consistent with model's own predictions
        static_idx = model_contact > 0.95  # N x S x 4
        model_feet = model_xp[:, :, foot_idx]  # foot positions (N, S, 4, 3)
        model_foot_v = torch.zeros_like(model_feet)
        model_foot_v[:, :-1] = (
            model_feet[:, 1:, :, :] - model_feet[:, :-1, :, :]
        )  # (N, S-1, 4, 3)
        model_foot_v[~static_idx] = 0
        foot_loss = self.loss_fn(
            model_foot_v, torch.zeros_like(model_foot_v), reduction="none"
        )
        foot_loss = reduce(foot_loss, "b ... -> b (...)", "mean")

        losses = (
            0.636 * loss.mean(),
            1.000 * loss_root.mean(),
            0.646 * fk_loss.mean(),
            10.942 * foot_loss.mean(),
            2.000 * inpaint_loss.mean() if self.model.use_masks else torch.tensor(0.0, device='cuda')
        )
        return sum(losses), losses

    def loss(self, x, cond, t_override=None, normalizer=None):
        batch_size = len(x)
        if t_override is None:
            # randomly generate t's
            t = torch.randint(0, self.n_timestep, (batch_size,), device=x.device).long()
        else:
            t = torch.full((batch_size,), t_override, device=x.device).long()
        return self.p_losses(x, cond, t, normalizer)

    def forward(self, x, cond, t_override=None, normalizer=None):
        return self.loss(x, cond, t_override, normalizer)