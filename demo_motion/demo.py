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
    
    if type(opt.constraint) == TrajectoryConstraint:
        opt.constraint.set_normalizer(model.normalizer)
    if type(opt.constraint) == EndEffectorConstraint or type(opt.constraint) == KineticEnergyConstraint:
        opt.constraint.set_normalizer(model.normalizer)
        opt.constraint.set_smpl(model.smpl)
    
    motion_dir = os.path.join(opt.motion_save_dir, f"{opt.model_name}_{opt.constraint}/{opt.method}")
    if not os.path.isdir(motion_dir): os.makedirs(motion_dir)
    samples_file = os.path.join(motion_dir, "normal_samples.pt")
    
    NUM = 1
    NUM_TIMESTEPS = 50
    print(f'Generating {NUM} normal sample{"" if NUM == 1 else "s"}...')
    shape = (NUM, model.horizon, model.repr_dim)
    extra_args = {}
    
    if opt.method == "dps":
        extra_args["weight"] = 0.5
        samples = model.diffusion.dps_sample(shape, sample_steps=NUM_TIMESTEPS, constraint_obj=opt.constraint, weight=extra_args["weight"])
    elif opt.method == "dsg":
        extra_args["gr"] = 0.1
        samples = model.diffusion.dsg_sample(shape, sample_steps=NUM_TIMESTEPS, constraint_obj=opt.constraint, gr=extra_args["gr"])
    elif opt.method == "trust":
        extra_args["norm_upper_bound"] = 80
        extra_args["iterations_max"] = 5
        extra_args["gradient_norm"] = 1
        extra_args["iteration_func"] = lambda time_next: 1 # 1
        model.diffusion.set_trust_parameters(iteration_func=extra_args["iteration_func"], norm_upper_bound=extra_args["norm_upper_bound"], iterations_max=extra_args["iterations_max"], gradient_norm=extra_args["gradient_norm"])
        samples, traj_found = model.diffusion.trust_sample(shape, constraint_obj=opt.constraint, debug=True)
        
    print(f'Finished generating trust samples.')
    if opt.save_motions: 
        torch.save(samples, samples_file)
        print(f'Saved in {motion_dir}')
    
    if not opt.no_render:
        print(f'Rendering {NUM} samples...')
        render_dir = os.path.join(opt.render_dir, f"{opt.model_name}_{opt.constraint}/{opt.method}")
        if not os.path.isdir(render_dir): os.makedirs(render_dir)
        just_render_simple(model.smpl, samples[:NUM], model.normalizer, render_out=render_dir, constraint=opt.constraint)
        
        # if opt.method == "trust":
        #     print(f'Rendering trajectory changes...')
        #     ani = trajectory_animation(traj_found, traj)
        #     ani.save(os.path.join(render_dir, 'trajectory.mp4'), writer='ffmpeg')
        
        print('Finished rendering samples.\n')
    
    # get_all_metrics(samples, opt.constraint, model, exp_name=f"{opt.model_name}_{opt.method}_{opt.constraint}")


if __name__ == "__main__":
    opt = parse_test_opt()
    opt.motion_save_dir = "./motions"
    opt.render_dir = "renders/experimental"
    opt.save_motions = False
    opt.no_render = False
    opt.predict_contact = True
    opt.checkpoint = "../runs/motion/exp4-train-4950.pt"
    opt.model_name = "fixes_4950"
    
    # x_traj = torch.cat((torch.linspace(X_START, 0.2, 30), torch.linspace(0.2, 0.2, 30)))
    # y_traj = torch.cat((torch.linspace(Y_START, Y_START, 30), torch.linspace(Y_START, 0.3, 30)))
    # traj = torch.stack((x_traj, y_traj)).T
    # const = TrajectoryConstraint(traj=traj)
    # const.set_name("Lshape")
    # opt.constraint = const
    
    # points = [(0, 4, X_START), (30, 4, X_START), (59, 4, X_START),
    #           (0, 5, Y_START), (30, 5, 0.3), (59, 5, Y_START)]
    # const = SpecifiedPointConstraint(points=points)
    # const.set_name("specified_up_and_back")
    # opt.constraint = const

    # points = [(25, 6, 0.3), (30, 6, 0.8), (35, 6, 0.3)]
    # const = SpecifiedPointConstraint(points=points)
    # const.set_name("specified_jump")
    # opt.constraint = const
    
    # points = [(0, "lwrist", 0.0, 0.0, 1.5), (30, "lwrist", 0.0, 1.0, 1.5), (59, "lwrist", 1.0, 1.0, 1.5)]
    # const = EndEffectorConstraint(points=points)
    # const.set_name("lwrist")
    # opt.constraint = const
    
    # points = [(0, "rankle", 0.0, 0.0, 0.0), (30, "rankle", 0.0, 1.0, 1.5)]
    # const = EndEffectorConstraint(points=points)
    # const.set_name("rankle")
    # opt.constraint = const
    
    const = KineticEnergyConstraint(KE=0)
    const.set_name("KE=0")
    opt.constraint = const
    
    for method in ["dps", "dsg", "trust"]:
        opt.method = method
        main(opt)