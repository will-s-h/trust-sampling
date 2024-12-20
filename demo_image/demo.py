## to be able to import all modules
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
from PIL import Image
from model.unet import create_model
from diffusion.diffusion import GaussianDiffusion
from constraints.SuperResolution import SuperResolutionConstraint
from constraints.Inpaint import InpaintConstraint
from constraints.GaussianDeblur import GaussianBlurConstraint
from constraints.Sketch import FaceSketchConstraint
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

def all_image_paths(directory):
    paths = []
    for dirpath, dirnames, files in os.walk(directory):
        for file in files:
            if file.endswith('.png') or file.endswith('.JPEG'):
                full_path = os.path.join(dirpath, file)
                paths.append(full_path)
    paths.sort()
    return paths

def main(args):
    model_config = load_yaml("./ffhq_model_config.yaml" if args.model == "ffhq" else "./imagenet_model_config.yaml")
    model = create_model(**model_config).to('cuda')

    all_paths = all_image_paths(args.dataset_path)
    batch_size = 1
    if args.constraint == "inpaint":
        masks = torch.load('../dataset/masks.pt')
    
    avg_nfes = 0
    
    if args.iterations_max == 'InverseScheduler':
        args.iterations_max = InverseScheduler(ddim_steps=200, nfes=1000)
    elif args.iterations_max == 'LinearScheduler':
        args.iterations_max = LinearScheduler(1, 8, 1000)
    elif args.iterations_max == 'LinearScheduler500':
        args.iterations_max = LinearScheduler(1, 4, 1000)
    elif args.iterations_max == 'StochasticLinearScheduler':
        args.iterations_max = StochasticLinearScheduler(0, 8)
    elif args.iterations_max == 'StochasticLinearScheduler_Other':
        args.iterations_max = StochasticLinearScheduler(2, 6)
    elif args.iterations_max == 'StochasticLinearScheduler500':
        args.iterations_max = StochasticLinearScheduler(0, 4)
    elif args.iterations_max == 'StochasticLinearScheduler500_Other':
        args.iterations_max = StochasticLinearScheduler(1, 3)
    elif args.iterations_max == 'SLS_0_3':
        args.iterations_max = StochasticLinearScheduler(0, 3)
    elif isinstance(args.iterations_max, str):
        args.iterations_max = int(args.iterations_max)
    
    if args.gradient_norm == 'InverseNormScheduler':
        args.gradient_norm = InverseNormScheduler(args.iterations_max)
    elif isinstance(args.gradient_norm, str):
        args.gradient_norm = int(args.gradient_norm)
    
    for j in range(len(all_paths) // batch_size):
        paths = all_paths[j * batch_size : (j+1) * batch_size]
        
        if args.constraint == "super_resolution":
            const = SuperResolutionConstraint(paths, scale_factor=4, noise=args.gaussian_noise)
        elif args.constraint == "inpaint":
            mask = masks[j * batch_size : (j+1) * batch_size].squeeze(1)
            const = InpaintConstraint(paths, mask=mask, noise=args.gaussian_noise)
        elif args.constraint == "gaussian_deblur":
            const = GaussianBlurConstraint(paths, 61, 3.0, noise=args.gaussian_noise)
            
        
        NUM_SAMPLES = len(paths)
        SHAPE = (NUM_SAMPLES, 3, 256, 256)
        diffusion = GaussianDiffusion(model, schedule="linear", n_timestep=1000, predict_epsilon=True, clip_denoised=True, learned_variance=True).to('cuda')
        extra_args = {}
        
        if args.method == "dps":
            SAMPLE_STEPS = 1000
            extra_args["weight"] = 1.0
            samples = diffusion.dps_sample(SHAPE, sample_steps=SAMPLE_STEPS, constraint_obj=const, weight=extra_args["weight"])
        elif args.method == "dsg":
            SAMPLE_STEPS = 1000
            extra_args["gr"] = 0.2
            extra_args["interval"] = 20
            samples = diffusion.dsg_sample(SHAPE, sample_steps=SAMPLE_STEPS, constraint_obj=const, gr=extra_args["gr"], interval=extra_args["interval"])
        elif args.method == "lgdmc":
            SAMPLE_STEPS = 1000
            extra_args["weight"] = 1.0
            extra_args["n"] = 10
            samples = diffusion.lgdmc_sample(SHAPE, sample_steps=SAMPLE_STEPS, constraint_obj=const, weight=extra_args["weight"], n=extra_args["n"])
        elif args.method == "trust":
            SAMPLE_STEPS = 200
            extra_args["norm_upper_bound"] = args.norm_upper_bound
            extra_args["iterations_max"] = args.iterations_max
            extra_args["gradient_norm"] = args.gradient_norm
            extra_args["iteration_func"] = lambda time_next: 1 # 1
            diffusion.set_trust_parameters(iteration_func=extra_args["iteration_func"], norm_upper_bound=extra_args["norm_upper_bound"], iterations_max=extra_args["iterations_max"], gradient_norm=extra_args["gradient_norm"])
            samples, nfes = diffusion.edited_trust_sample(SHAPE, sample_steps=SAMPLE_STEPS, constraint_obj=const, debug=True)
            avg_nfes += torch.mean(nfes).item() / (len(all_paths) // batch_size)

        # plot all experiments
        
        save_dir = f"{args.save_dir}/{const}/{args.dataset_name}_{args.method}"
        if args.method == "trust":
            save_dir += f"({args.norm_upper_bound},{args.iterations_max},{args.gradient_norm})"
        if not os.path.exists(save_dir): os.makedirs(save_dir)

        for i, sample in enumerate(samples):
            plt.imsave(f'{save_dir}/result{j*batch_size + i:05}.png', clear_color(sample))
    
    if args.method == "trust":
        print(f'average NFEs: {avg_nfes}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="dsg")
    parser.add_argument("--model", type=str, default="ffhq")
    parser.add_argument("--constraint", type=str, default="super_resolution")
    parser.add_argument("--dataset_path", type=str, default="../dataset/ffhq256-10")
    parser.add_argument("--dataset_name", type=str, default="ffhq")
    parser.add_argument("--norm_upper_bound", type=float, default=440)
    parser.add_argument("--iterations_max", default=4)
    parser.add_argument("--gradient_norm", default=1)
    parser.add_argument("--gaussian_noise", default=0)
    parser.add_argument("--save_dir", default='./results')
    args = parser.parse_args()
    main(args)