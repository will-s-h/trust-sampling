## to be able to import all modules
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

## other imports
from motion_args import parse_test_opt
from model.motion_wrapper import MotionWrapper
import torch
import os
from visualization.render_motion import trajectory_animation, just_render_simple
from constraints.trajectory_constraint import TrajectoryConstraint
from constraints.specified_points import SpecifiedPointConstraint
from constraints.end_effector import EndEffectorConstraint
from constraints.kinetic_energy import KineticEnergyConstraint
from constraints.combine import Combine
from evaluator import get_all_metrics

X_START, Y_START = -0.107, -0.1545
MODEL_PREV_CKPT = None
model = None

def main(opt):
    global MODEL_PREV_CKPT
    global model

    if MODEL_PREV_CKPT != opt.checkpoint:
        MODEL_PREV_CKPT = opt.checkpoint
        print('**********************')
        print('Loading model...')
        model = MotionWrapper(opt.checkpoint, predict_contact=opt.predict_contact)
        model.eval()
        print('Model loaded.')
        print('**********************\n')
    
    NUM = 1
    NUM_TIMESTEPS = 50
    print(f'Generating {NUM} normal sample{"" if NUM == 1 else "s"}...')
    shape = (NUM, model.horizon, model.repr_dim)
    samples = model.diffusion.ddim_sample(shape, sample_steps=NUM_TIMESTEPS)
        
    print(f'Finished generating unconstrained samples.')
    motion_dir = os.path.join(opt.motion_save_dir, f"unconstrained")
    if opt.save_motions: 
        if not os.path.isdir(motion_dir): os.makedirs(motion_dir)
        samples_file = os.path.join(motion_dir, "normal_samples.pt")
        torch.save(samples, samples_file)
        print(f'Saved in {motion_dir}')
    
    if not opt.no_render:
        print(f'Rendering {NUM} samples...')
        render_dir = os.path.join(opt.render_dir, f"unconstrained")
        if not os.path.isdir(render_dir): os.makedirs(render_dir)
        just_render_simple(model.smpl, samples[:NUM], model.normalizer, render_out=render_dir)
        print('Finished rendering samples.\n')


if __name__ == "__main__":
    opt = parse_test_opt()
    opt.motion_save_dir = "./motions"
    opt.render_dir = "renders/"
    opt.save_motions = False
    opt.no_render = False
    opt.predict_contact = True
    opt.checkpoint = "../runs/motion/exp4-train-4950.pt"
    opt.model_name = "fixes_4950"
    
    for method in ["dps", "dsg", "trust"][2:]:
        opt.method = method
        main(opt)