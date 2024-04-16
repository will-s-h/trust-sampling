## to be able to import all modules
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

import os
import torch
from constraints.line import LineConstraint
from constraints.spiral import SpiralConstraint
from constraints.cubic import CubicConstraint
from model.model_2D import MLP
from diffusion.diffusion import GaussianDiffusion
from visualization.render_2D import *
from evaluator import get_all_metrics

# the spiral dataset uses r=theta/3π for 2π ≤ theta ≤ 6π
MODEL_NAME = "spiral_base"
model = MLP()
path = f"../runs/2D/{MODEL_NAME}/model.pth"
exp_name = f"{MODEL_NAME}_dsg_gr1"
model.load_state_dict(torch.load(path))
distr = "spiral"
distribution = SpiralConstraint()

# let's define a constraint f(x,y) = 2x+y+1 = 0
# possible constraints: LineConstraint(2, 1, -1), LineConstraint(1, 2, 0), CubicConstraint(1, 0, -1, 0)
constraint = CubicConstraint(1, 0, -1, 0)

# is this an experimental version: if so, save in special experimental path
EXPERIMENTAL = True

# basic parameters
NUM_TIMESTEPS = 50
NUM_SAMPLES = 1000
ADD_NOISE = True

diffusion = GaussianDiffusion(model, schedule="linear", n_timestep=NUM_TIMESTEPS, predict_epsilon=True, clip_denoised=False)
# diffusion.set_trust_parameters(iteration_func=lambda x: 1, norm_upper_bound=35, iterations_max=1, gradient_norm=1)
# diffusion.set_trust_parameters(iteration_func=lambda x: 40 if 20 <= x <= 30 else 1, norm_upper_bound=35, iterations_max=2, gradient_norm=1)
# samples = diffusion.trust_sample((NUM_SAMPLES, 2), sample_steps=NUM_TIMESTEPS, constraint_obj=constraint, save_intermediates=True)
# samples = diffusion.dps_sample((NUM_SAMPLES, 2), sample_steps=NUM_TIMESTEPS, constraint_obj=constraint, weight=2, save_intermediates=True)
samples = diffusion.dsg_sample((NUM_SAMPLES, 2), sample_steps=NUM_TIMESTEPS, constraint_obj=constraint, gr=1, save_intermediates=True)

samples = [sample.numpy() for sample in samples]

# plot all experiments
def sanity_check(samples):
    samples = samples[-1]
    num_reasonable = sum((samples[:, 0] <= 2) & (samples[:, 0] >= -2) & (samples[:, 1] <= 2) & (samples[:, 1] >= -2))
    print(f'points within reasonable bounds: {num_reasonable}')

sanity_check(samples)

save_dir = f"plots/{exp_name}"
if not os.path.exists(save_dir): os.makedirs(save_dir)
print(f'Saving all plots in path {save_dir}.')

fig, ax = distribution_metric(samples, model)
fig.savefig(f"{save_dir}/distribution_metric.png", facecolor='white', dpi=300)
print('Saved distribution_metric.png')

fig, ax = constraint_metric(samples, constraint)
fig.savefig(f"{save_dir}/constraint_metric.png", facecolor='white', dpi=300)
print('Saved constraint_metric.png')

fig, ax = plot_constraint_exp(samples, constraint, distr=distr)
fig.savefig(f"{save_dir}/constraint_exp.png", facecolor="white", dpi=300)
print('Saved constraint_exp.png.')

# fig, ax = plot_diffusion_steps(samples, distribution)
# fig.savefig(f"{save_dir}/diffusion_steps.png", facecolor="white", dpi=300)
# print('Saved diffusion_steps.png.')

ani = video_all_steps(samples, constraint, distr=distr)
ani.save(f"{save_dir}/diffusion.mp4", writer='ffmpeg')
print('Saved diffusion.mp4.')

get_all_metrics(samples, constraint, distribution, exp_name=exp_name, auto_save=True, file_path="all_2D_experiments.csv")