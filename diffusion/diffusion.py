import copy
from functools import partial
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .utils import extract, make_beta_schedule


def generate_sequence(n, total_sum):
    # Initial parameters
    a = 1
    d = (total_sum - n * a) / (n * (n - 1) / 2)

    # Generate the sequence
    sequence = [int(round(a + i * d)) for i in range(n)]

    # Adjust the sequence to ensure the sum is exactly 200
    current_sum = sum(sequence)
    difference = total_sum - current_sum

    # Distribute the difference
    i = 0
    while difference != 0:
        if difference > 0:
            sequence[i] += 1
            difference -= 1
        elif difference < 0:
            sequence[i] -= 1
            difference += 1
        i = (i + 1) % n

    # turn sequence into tensor
    sequence = torch.tensor(sequence, dtype=torch.float32)

    return sequence






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
        n_timestep=1000,
        schedule="linear",
        predict_epsilon=False,
        clip_denoised=True,
        learned_variance=False
    ):
        super().__init__()
        self.model = model  # this model is the either a MotionDecoder or 2D model
        self.ema = EMA(0.9999)
        self.master_model = copy.deepcopy(self.model)
        
        betas = torch.Tensor(
            make_beta_schedule(schedule=schedule, n_timestep=n_timestep)
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timestep = int(n_timestep)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon
        self.learned_variance = learned_variance

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
        
        # version used by image model
        self.register_buffer(
            "posterior_log_variance_clipped_",
            torch.log(torch.cat((posterior_variance[1].unsqueeze(0), posterior_variance[1:])))
        )
        
        self.register_buffer(
            "posterior_mean_coef1",
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
        )
    # ------------------------------------------ sampling ------------------------------------------#
    
    def predict_start_from_noise(self, x_t, t, noise):
        """
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        """
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
        
    # predict epsilon from x0
    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    # predict epsilon and x0 from x_t
    def model_predictions(self, x, t, clip_x_start = False):
        model_output = self.model(x, t)

        # In the case of "learned" variance, model will give twice channels.
        if model_output.shape[1] == 2 * x.shape[1] and self.learned_variance:
            model_output, model_var_values = torch.split(model_output, x.shape[1], dim=1)
            min_log = extract(self.posterior_log_variance_clipped_, t, x.shape)
            max_log = extract(torch.log(self.betas), t, x.shape)
            frac = (model_var_values + 1.0) / 2.0
            model_log_variance = frac * max_log + (1 - frac) * min_log
        
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity
        
        if self.predict_epsilon:
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)
        else:
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return pred_noise, x_start, model_log_variance if self.learned_variance else pred_noise, x_start
        
    @torch.no_grad()
    def ddim_sample(self, shape, sample_steps=50, save_intermediates=False, **kwargs):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.n_timestep, sample_steps, 1

        # times = torch.linspace(-1, sampling_timesteps-1, steps=sampling_timesteps +1)
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn(shape, device = device)
        x_start = None
        intermediates = []
        if save_intermediates: intermediates.append(x)

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            # produce a bunch of times, for each of the samples in the batch
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            # predict completely denoised product
            pred_noise, x_start, *_ = self.model_predictions(x, time_cond, clip_x_start = self.clip_denoised)
            if self.learned_variance:
                pred_log_var = _[0]

            if time_next < 0:
                x = x_start
                continue
            
            # apply diffusion noise again, except with one less step of noise than what was denoised.
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            
            alpha_one = self.alphas_cumprod[time-1]
            sigma_one = eta * ((1 - alpha / alpha_one) * (1 - alpha_one) / (1 - alpha)).sqrt()

            noise = torch.randn_like(x)
            # model_mean = x_start * alpha_next.sqrt() + c * pred_noise
            # new_pred_noise, *_ = self.model_predictions(model_mean, time_cond, clip_x_start=self.clip_denoised)
            # norm_noise = torch.norm(new_pred_noise, dim=(-2, -1)).cpu()
            # print(norm_noise)
            x = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise #(sigma / sigma_one) * torch.exp(0.5 * pred_log_var) * noise if self.learned_variance else sigma * noise
            
            if save_intermediates: intermediates.append(x)
                  
            new_pred_noise, *_ = self.model_predictions(x, time_cond, clip_x_start=self.clip_denoised)
            print(f'reached norm: {torch.norm(new_pred_noise).item()}')
        return intermediates if save_intermediates else x
    
    def _get_trajectory(self, x):
        return x[..., 4:6].squeeze(0).cpu() if x.dim() == 3 and x.shape[0] == 1 else x[..., 4:6].cpu()

    def get_norm_upper_bound(self):
        raise NotImplementedError("need to implement automatic norm upper bound finding")

    @torch.no_grad()  # note that weight ranges from 0.3 to 1 in DPS paper
    def dps_sample(self, shape, sample_steps=50, constraint_obj=None, weight=1, save_intermediates=False, **kwargs):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.n_timestep, sample_steps, 1

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn(shape, device = device)
        x_start = None
        intermediates = []
        if save_intermediates: intermediates.append(x)

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            # produce a bunch of times, for each of the samples in the batch
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            
            # predict completely denoised product
            pred_noise, x_start, *_ = self.model_predictions(x, time_cond, clip_x_start = self.clip_denoised)

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
                  
            # DPS gradient correction:
            g = constraint_obj.gradient(x_start, lambda x: self.model_predictions(x, time_cond, clip_x_start=self.clip_denoised))
            norms = torch.norm(g.view(g.shape[0], -1), dim=1).view((g.shape[0],) + tuple([1 for _ in range(len(g.shape[1:]))])).expand(g.shape) + 1e-6  #avoid div by 0
            g *= weight / norms
            x += g  # note that the negative sign is included in the .gradient function, in our formulation
            
            if save_intermediates: intermediates.append(x)
            
        return intermediates if save_intermediates else x
    
    @torch.no_grad()  # in paper, gr=0.1 or 0.2
    def dsg_sample(self, shape, sample_steps=50, constraint_obj=None, gr=0.1, save_intermediates=False, **kwargs):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.n_timestep, sample_steps, 1

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn(shape, device = device)
        x_start = None
        intermediates = []
        if save_intermediates: intermediates.append(x)

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            # produce a bunch of times, for each of the samples in the batch
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            
            # predict completely denoised product
            pred_noise, x_start, *_ = self.model_predictions(x, time_cond, clip_x_start = self.clip_denoised)

            if time_next < 0:
                x = x_start
                continue
            
            # apply diffusion noise again, except with one less step of noise than what was denoised.
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x)

            x_next_mean = x_start * alpha_next.sqrt() + c * pred_noise
            x = x_next_mean + sigma * noise
                  
            # DSG sizing of gradient: sqrt(n) * sigma_t
            g = constraint_obj.gradient(x_start, lambda x: self.model_predictions(x, time_cond, clip_x_start=self.clip_denoised))
            norms = torch.norm(g.view(g.shape[0], -1), dim=1).view((g.shape[0],) + tuple([1 for _ in range(len(g.shape[1:]))])).expand(g.shape) + 1e-6  #avoid div by 0
            g *= sigma * x.numel() ** 0.5 / norms
            x_mod = x_start*alpha_next.sqrt() + c*pred_noise + g  # note that g already has - sign within it
            
            # weighted average between xmod and x for diversity
            x = x + gr*(x_mod - x)
            
            if save_intermediates: intermediates.append(x)
            
        return intermediates if save_intermediates else x
    
    def set_trust_parameters(self, iteration_func = None, norm_upper_bound = None, iterations_max = 1, gradient_norm = 1, refine = True):
        self.iteration_func = iteration_func if iteration_func is not None else (lambda x: 1)
        self.norm_upper_bound = norm_upper_bound if norm_upper_bound is not None else self.get_norm_upper_bound()
        self.iterations_max = iterations_max
        self.gradient_norm = gradient_norm
        self.refine = refine

    @torch.no_grad()
    def trust_sample_Jschedule(self, shape, sample_steps=50, constraint_obj=None, save_intermediates=False, debug=False,
                     **kwargs):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[
            0], self.betas.device, self.n_timestep, sample_steps, 1
        assert constraint_obj is not None, "must pass in constraint object!"
        assert (
                    not save_intermediates or not debug), "cannot both save intermediates and be in debug mode. must pick one of the two!"

        times = torch.linspace(-1, total_timesteps - 1,
                               steps=sampling_timesteps + 1)  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn(shape, device=device)

        x_start = None
        traj, intermediates = [], []
        if debug: traj.append((self._get_trajectory(x), 1001, 'starting trajectory'))
        if save_intermediates: intermediates.append(x)

        neural_function_evals = torch.zeros(x.shape[0], dtype=torch.float)
        J_schedule_counter = 0
        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            iterations = self.iteration_func(time_next)
            self.current_iterations_max = self.iterations_max[J_schedule_counter]
            J_schedule_counter += 1
            for _ in range(iterations):

                time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
                pred_noise, x_start, *_ = self.model_predictions(x, time_cond, clip_x_start=self.clip_denoised)
                neural_function_evals += 1

                if debug: traj.append((self._get_trajectory(x_start), time, 'diffusion step'))

                if time_next < 0:  # we are in the last step
                    pred_noise, *_ = self.model_predictions(x_start, time_cond, clip_x_start=self.clip_denoised)
                    # print(f'final norm: {torch.norm(pred_noise).item()}')
                    if save_intermediates: intermediates.append(x_start)
                    x = x_start
                    break

                # apply diffusion noise again, except with one less step of noise than what was denoised.
                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]
                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()

                model_mean = x_start * alpha_next.sqrt() + c * pred_noise
                new_pred_noise, *_ = self.model_predictions(model_mean, time_cond, clip_x_start=self.clip_denoised)
                gradient_iterations = torch.zeros((batch, 1))
                norm_noise = torch.norm(new_pred_noise, dim=(-2, -1)).cpu()
                grad_step = (gradient_iterations < self.current_iterations_max).squeeze() * (
                            norm_noise <= self.norm_upper_bound)

                while grad_step.any():
                    g = constraint_obj.gradient(model_mean, lambda x: self.model_predictions(x, time_cond,
                                                                                             clip_x_start=self.clip_denoised))
                    norms = torch.norm(g.view(g.shape[0], -1), dim=1).view(
                        (g.shape[0],) + tuple([1 for _ in range(len(g.shape[1:]))])).expand(
                        g.shape) + 1e-6  # avoid div by 0
                    g *= self.gradient_norm / norms
                    model_mean = model_mean + grad_step.unsqueeze(-1).unsqueeze(-1).to('cuda') * g
                    if debug: traj.append((self._get_trajectory(model_mean), time, 'gradient step'))
                    gradient_iterations += grad_step.unsqueeze(-1)
                    new_pred_noise, *_ = self.model_predictions(model_mean, time_cond, clip_x_start=self.clip_denoised)
                    neural_function_evals += grad_step
                    norm_noise = torch.norm(new_pred_noise, dim=(-2, -1)).cpu()
                    grad_step = (gradient_iterations < self.current_iterations_max).squeeze() * (
                            norm_noise <= self.norm_upper_bound)

                noise = torch.randn_like(x)
                x = model_mean + sigma * noise

                if debug: traj.append((self._get_trajectory(x), time, 'noise'))
            if save_intermediates: intermediates.append(x)

        print(f"neural evaluations: {neural_function_evals}")
        print(constraint_obj.constraint(x))
        # long_sample = constraint_obj.normalizer.unnormalize(constraint_obj.stack_samples(x))
        # long_sample_before = constraint_obj.normalizer.unnormalize(constraint_obj.stack_samples(x_start_before))
        if debug: return (x, traj, neural_function_evals)
        if save_intermediates: return intermediates

        return x


    @torch.no_grad()
    def trust_sample(self, shape, sample_steps=50, constraint_obj=None, save_intermediates=False, debug=False, **kwargs):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.n_timestep, sample_steps, 1
        assert constraint_obj is not None, "must pass in constraint object!"
        assert (not save_intermediates or not debug), "cannot both save intermediates and be in debug mode. must pick one of the two!"

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        
        x = torch.randn(shape, device = device)

        x_start = None
        traj, intermediates = [], []
        if debug: traj.append((self._get_trajectory(x), 1001, 'starting trajectory'))
        if save_intermediates: intermediates.append(x)
        
        neural_function_evals = torch.zeros(x.shape[0], dtype=torch.float)

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            iterations = self.iteration_func(time_next)
            for _ in range(iterations):

                time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
                pred_noise, x_start, *_ = self.model_predictions(x, time_cond, clip_x_start = self.clip_denoised)
                neural_function_evals += 1

                if debug: traj.append((self._get_trajectory(x_start), time, 'diffusion step'))

                if time_next < 0: # we are in the last step
                    if self.refine :
                        nr_refinement_steps = 0
                        while neural_function_evals < 201:
                            print('loss:', constraint_obj.constraint(x_start)[0].item(),'loss2:', constraint_obj.constraint(x_start)[1].item(),'loss3:', constraint_obj.constraint(x_start)[2].item())
                            # model_mean = x_start * alpha_next.sqrt() + c * pred_noise
                            g = constraint_obj.gradient(x, lambda x: self.model_predictions(x, time_cond,
                                                                                                    clip_x_start=self.clip_denoised))
                            norms = torch.norm(g.view(g.shape[0], -1), dim=1).view(
                                (g.shape[0],) + tuple([1 for _ in range(len(g.shape[1:]))])).expand(
                                g.shape) + 1e-6  # avoid div by 0
                            g *= self.gradient_norm / norms
                            # noise = torch.randn_like(x)
                            # x = model_mean + 0.1 * grad_step.unsqueeze(-1).unsqueeze(-1).to('cuda') * g  #+ sigma * noise.
                            x = x + 0.1 * g
                            pred_noise, x_start, *_ = self.model_predictions(x, time_cond,
                                                                             clip_x_start=self.clip_denoised)
                            neural_function_evals += 1
                            nr_refinement_steps += 1
                        if save_intermediates: intermediates.append(x_start)
                        print('loss:', constraint_obj.constraint(x_start)[0].item(), 'loss2:',
                              constraint_obj.constraint(x_start)[1].item())
                        x = x_start
                        break
                    else:
                        pred_noise, *_ = self.model_predictions(x_start, time_cond, clip_x_start=self.clip_denoised)
                        # print(f'final norm: {torch.norm(pred_noise).item()}')
                        if save_intermediates: intermediates.append(x_start)
                        x = x_start
                        break
                
                # apply diffusion noise again, except with one less step of noise than what was denoised.
                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]
                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()

                model_mean = x_start * alpha_next.sqrt() + c * pred_noise
                new_pred_noise, *_ = self.model_predictions(model_mean, time_cond, clip_x_start=self.clip_denoised)
                gradient_iterations = torch.zeros((batch, 1))
                norm_noise = torch.norm(new_pred_noise, dim=(-2, -1)).cpu()
                grad_step = (gradient_iterations < self.iterations_max).squeeze() * (norm_noise <= self.norm_upper_bound)

                while grad_step.any():
                    g = constraint_obj.gradient(model_mean, lambda x: self.model_predictions(x, time_cond, clip_x_start=self.clip_denoised))
                    norms = torch.norm(g.view(g.shape[0], -1), dim=1).view((g.shape[0],) + tuple([1 for _ in range(len(g.shape[1:]))])).expand(g.shape) + 1e-6  #avoid div by 0
                    g *= self.gradient_norm / norms
                    model_mean = model_mean + grad_step.unsqueeze(-1).unsqueeze(-1).to('cuda') * g
                    if debug: traj.append((self._get_trajectory(model_mean), time, 'gradient step'))
                    gradient_iterations += grad_step.unsqueeze(-1)
                    new_pred_noise, *_ = self.model_predictions(model_mean, time_cond, clip_x_start=self.clip_denoised)
                    neural_function_evals += grad_step
                    norm_noise = torch.norm(new_pred_noise, dim=(-2, -1)).cpu()
                    grad_step = (gradient_iterations < self.iterations_max).squeeze() * (
                                norm_noise <= self.norm_upper_bound)

                noise = torch.randn_like(x)
                x = model_mean + sigma * noise

                if debug: traj.append((self._get_trajectory(x), time, 'noise'))
            if save_intermediates: intermediates.append(x)
        
        print(f"neural evaluations: {neural_function_evals}")
        print(constraint_obj.constraint(x))
        # long_sample = constraint_obj.normalizer.unnormalize(constraint_obj.stack_samples(x))
        # long_sample_before = constraint_obj.normalizer.unnormalize(constraint_obj.stack_samples(x_start_before))
        if debug: return (x, traj, neural_function_evals)
        if save_intermediates: return intermediates

        return x


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