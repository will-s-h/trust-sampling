## to be able to import all modules
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
# .resolve is necessary for DPS environment for some reason

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
from model.unet import create_model
from diffusion.diffusion import GaussianDiffusion
from diffusion.iteration_schedule import *

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

def main(args):
    model_config = load_yaml("/move/u/willsh/GitHub/trust-sampling/demo_image/ffhq_model_config.yaml" if args.model == "ffhq" else "./imagenet_model_config.yaml")
    model = create_model(**model_config).to('cuda')
    model.eval()

    batch_size = 1
    
    for j in range(args.n_samples // batch_size):
        SAMPLE_STEPS = 1000
        NUM_SAMPLES = batch_size
        SHAPE = (NUM_SAMPLES, 3, 256, 256)
        diffusion = GaussianDiffusion(model, schedule="linear", n_timestep=1000, predict_epsilon=True, clip_denoised=True, learned_variance=True).to('cuda')
        samples = diffusion.ddim_sample(SHAPE, SAMPLE_STEPS)

        # plot all experiments
        save_dir = f"./ddim"
        if not os.path.exists(save_dir): os.makedirs(save_dir)

        for i, sample in enumerate(samples):
            plt.imsave(f'{save_dir}/result{j*batch_size + i:05}.png', clear_color(sample))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="dps")
    parser.add_argument("--model", type=str, default="ffhq")
    parser.add_argument("--n_samples", type=int, default=10)
    args = parser.parse_args()
    main(args)