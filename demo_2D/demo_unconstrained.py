## to be able to import all modules
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

import os
import torch
from model.model_2D import MLP
from diffusion.diffusion import GaussianDiffusion
from visualization.render_2D import video_all_steps

# the spiral dataset uses r=theta/3π for 2π ≤ theta ≤ 6π
MODEL_NAME = "spiral_base"
distr = 'spiral'
model = MLP()
path = f"../runs/2D/{MODEL_NAME}/model.pth"
model.load_state_dict(torch.load(path))

# perform experiments without constraint
NUM_TIMESTEPS = 50
NUM_SAMPLES = 1000
ADD_NOISE = True

diffusion = GaussianDiffusion(model, schedule="linear", n_timestep=NUM_TIMESTEPS, predict_epsilon=True, clip_denoised=False)
samples = diffusion.ddim_sample((NUM_SAMPLES, 2), sample_steps=NUM_TIMESTEPS, save_intermediates=True)

# plot all experiments
save_dir = f"plots/{MODEL_NAME}_t{NUM_TIMESTEPS}{'_wnoise' if ADD_NOISE else ''}"
if not os.path.exists(save_dir): os.makedirs(save_dir)

ani = video_all_steps(samples, None, distr=distr)
ani.save(f"{save_dir}/diffusion.mp4", writer='ffmpeg')
print('Saved diffusion.mp4.')