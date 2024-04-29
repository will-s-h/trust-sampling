## to be able to import all modules
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
from model.unet import create_model
from diffusion.diffusion import GaussianDiffusion
from constraints.SuperResolution import SuperResolutionConstraint

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def clear_color(x):
    if torch.is_complex(x):
        x = torch.abs(x)
    x = x.detach().cpu().squeeze().numpy()
    return normalize_np(np.transpose(x, (1, 2, 0)))

def normalize_np(img):
    """ Normalize img in arbitrary range to [0, 1] """
    img -= np.min(img)
    img /= np.max(img)
    return img

def all_image_paths(directory):
    paths = []
    for dirpath, dirnames, files in os.walk(directory):
        for file in files:
            if file.endswith('.png') or file.endswith('.JPEG'):
                full_path = os.path.join(dirpath, file)
                paths.append(full_path)
    paths.sort()
    return paths

def main(dataset = "ffhq", method = "trust"):
    model_config = load_yaml(f"./{dataset}_model_config.yaml")
    model = create_model(**model_config).to('cuda')

    paths = all_image_paths('../dataset/ffhq256-100')[:4]
    print(paths)
    const = SuperResolutionConstraint(paths, scale_factor=4)
    SAMPLE_STEPS = 200
    NUM_SAMPLES = len(paths)
    SHAPE = (NUM_SAMPLES, 3, 256, 256)
    diffusion = GaussianDiffusion(model, schedule="linear", n_timestep=1000, predict_epsilon=True, clip_denoised=True, learned_variance=True).to('cuda')
    extra_args = {}
    
    if method == "dps":
        extra_args["weight"] = 1
        samples = diffusion.dps_sample(SHAPE, sample_steps=SAMPLE_STEPS, constraint_obj=const, weight=extra_args["weight"])
    elif method == "dsg":
        extra_args["gr"] = 0.1
        samples = diffusion.dsg_sample(SHAPE, sample_steps=SAMPLE_STEPS, constraint_obj=const, gr=extra_args["gr"])
    elif method == "trust":
        extra_args["norm_upper_bound"] = 440 * (len(paths)) ** 0.5
        extra_args["iterations_max"] = 5
        extra_args["gradient_norm"] = 1
        extra_args["iteration_func"] = lambda time_next: 1 # 1
        diffusion.set_trust_parameters(iteration_func=extra_args["iteration_func"], norm_upper_bound=extra_args["norm_upper_bound"], iterations_max=extra_args["iterations_max"], gradient_norm=extra_args["gradient_norm"])
        samples = diffusion.trust_sample(SHAPE, sample_steps=SAMPLE_STEPS, constraint_obj=const)

    # plot all experiments
    save_dir = f"plots/{const}/{dataset}_{method}"
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    for i, sample in enumerate(samples):
        plt.imsave(f'{save_dir}/result{i:05}.png', clear_color(sample))

if __name__ == "__main__":
    main(dataset="ffhq", method="trust")