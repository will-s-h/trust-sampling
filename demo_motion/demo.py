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

X_START, Y_START = -0.107, -0.1545

def main(opt):
    print('**********************')
    print('Loading model...')
    model = MotionWrapper(opt.checkpoint, predict_contact=opt.predict_contact)
    model.eval()
    print('Model loaded.')
    print('**********************\n')
    
    motion_dir = os.path.join(opt.motion_save_dir, f"{opt.model_name}")
    if not os.path.isdir(motion_dir): os.makedirs(motion_dir)
    samples_file = os.path.join(motion_dir, "normal_samples.pt")
    
    NUM = 1
    print(f'Generating {NUM} normal sample{"" if NUM == 1 else "s"}...')
    
    x_traj = torch.linspace(X_START, 0.5, 60) #torch.cat((torch.linspace(X_START, 0.3, 30), torch.linspace(0.3, 0.3, 30)))
    y_traj = torch.linspace(Y_START, Y_START, 60) #torch.cat((torch.linspace(Y_START, Y_START, 30), torch.linspace(Y_START, 0.3, 30)))
    traj = torch.stack((x_traj, y_traj)).T
    const = TrajectoryConstraint(traj=traj)
    
    shape = (NUM, model.horizon, model.repr_dim)
    model.diffusion.set_trust_parameters(iteration_func=lambda time_next: 5 if time_next < 0 else 1, norm_upper_bound=80, iterations_max=5, gradient_norm=1)
    samples, traj_found = model.diffusion.trust_sample(shape, constraint_obj=const, debug=True)
    
    print(f'Finished generating trust samples.')
    if opt.save_motions: 
        torch.save(samples, samples_file)
        print(f'Saved in {motion_dir}')
    
    if not opt.no_render:
        print(f'Rendering {NUM} samples...')
        render_dir = os.path.join(opt.render_dir, f"{opt.model_name}")
        if not os.path.isdir(render_dir): os.makedirs(render_dir)
        just_render_simple(model.smpl, samples[:NUM], model.normalizer, render_out=render_dir)
        
        print(f'Rendering trajectory changes...')
        ani = trajectory_animation(traj_found, traj)
        ani.save(os.path.join(render_dir, 'trajectory.mp4'), writer='ffmpeg')
        
        print('Finished rendering samples.\n')


if __name__ == "__main__":
    opt = parse_test_opt()
    opt.save_motions = True
    opt.no_render = False
    opt.predict_contact = True
    opt.checkpoint = "../runs/motion/exp4-train-4950.pt"
    opt.model_name = "fixes_4950"
    main(opt)