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

device = torch.device('cuda')
model_config = load_yaml("./model_config.yaml")
model = create_model(**model_config).to('cuda')

# perform experiments without constraint
NUM_TIMESTEPS = 1000
NUM_SAMPLES = 1000
SHAPE = (1, 3, 256, 256)
diffusion = GaussianDiffusion(model, schedule="linear", n_timestep=NUM_TIMESTEPS, predict_epsilon=True, clip_denoised=True, learned_variance=True).to('cuda')
samples = diffusion.ddim_sample(SHAPE, sample_steps=NUM_TIMESTEPS, save_intermediates=False, device=torch.device('cuda'))

# plot all experiments
save_dir = f"plots/unconstrained"
if not os.path.exists(save_dir): os.makedirs(save_dir)

plt.imsave(f'{save_dir}/result.png', clear_color(samples))