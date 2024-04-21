## to be able to import all modules
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

import os
import torch
from constraints.line import LineConstraint
from constraints.spiral import SpiralConstraint
from constraints.cubic import CubicConstraint
from constraints.circle import CircleInequality
from constraints.polylike import PolyLikeConstraint
from constraints.combine import Combine
from model.model_2D import MLP
from diffusion.diffusion import GaussianDiffusion
from visualization.render_2D import *
from evaluator import sanity_check, get_all_metrics

MODEL_NAME_PREV = None
model = MLP()

def all_exps_constrained(MODEL_NAME, constraint, method, distr, distribution):
    global MODEL_NAME_PREV
    global model
    
    if MODEL_NAME_PREV != MODEL_NAME:
        path = f"../runs/2D/{MODEL_NAME}/model.pth"
        model.load_state_dict(torch.load(path))
        MODEL_NAME_PREV = MODEL_NAME
    exp_name = f"{MODEL_NAME}_{method}_{constraint}"

    # basic parameters
    NUM_TIMESTEPS = 50
    NUM_SAMPLES = 1000
    ADD_NOISE = True
    diffusion = GaussianDiffusion(model, schedule="linear", n_timestep=NUM_TIMESTEPS, predict_epsilon=True, clip_denoised=False)
    extra_args = {}

    if method == "dps":
        extra_args["weight"] = 0.1
        samples = diffusion.dps_sample((NUM_SAMPLES, 2), sample_steps=NUM_TIMESTEPS, constraint_obj=constraint, weight=extra_args["weight"], save_intermediates=True)
    elif method == "dsg":
        extra_args["gr"] = 0.1
        samples = diffusion.dsg_sample((NUM_SAMPLES, 2), sample_steps=NUM_TIMESTEPS, constraint_obj=constraint, gr=extra_args["gr"], save_intermediates=True)
    elif method == "trust":
        extra_args["norm_upper_bound"] = 35
        extra_args["iterations_max"] = 5
        extra_args["gradient_norm"] = 1
        extra_args["iteration_func"] = lambda x: 2
        diffusion.set_trust_parameters(iteration_func=extra_args["iteration_func"], norm_upper_bound=extra_args["norm_upper_bound"], 
                                    iterations_max=extra_args["iterations_max"], gradient_norm=extra_args["gradient_norm"])
        # diffusion.set_trust_parameters(iteration_func=lambda x: 40 if 20 <= x <= 30 else 1, norm_upper_bound=35, iterations_max=2, gradient_norm=1)
        samples = diffusion.trust_sample((NUM_SAMPLES, 2), sample_steps=NUM_TIMESTEPS, constraint_obj=constraint, save_intermediates=True)

    samples = [sample.numpy() for sample in samples]

    ##########################
    #    Plot experiments    #
    ##########################

    sanity_check(samples)

    save_dir = f"plots/{MODEL_NAME}_{constraint}/{method}"
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

    ani = video_all_steps(samples, constraint, distr=distr)
    ani.save(f"{save_dir}/diffusion.mp4", writer='ffmpeg')
    print('Saved diffusion.mp4.')

    get_all_metrics(samples, constraint, distribution, exp_name=exp_name, auto_save=True, file_path="all_2D_experiments.csv", **extra_args)
    
if __name__ == "__main__":
    # parameters
    MODEL_NAME = "spiral_base"
    constraint = Combine(CircleInequality(1.5, 0, 1, 2), CircleInequality(1, -1, 1, 1))
    distr = "spiral"
    distribution = SpiralConstraint()
    
    for method in ["dps", "dsg", "trust"]:
        all_exps_constrained(MODEL_NAME, constraint, method, distr, distribution)
        print()