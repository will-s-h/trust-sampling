## to be able to import all modules
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

## other imports
from motion_args import parse_test_opt
from model.motion_wrapper import MotionWrapper
import torch
import os
main_path = os.path.dirname(os.path.dirname(__file__))
from visualization.render_motion import render_to_compare
from constraints.trajectory_constraint_3d import TrajectoryConstraint3D
from constraints.specified_points import SpecifiedPointConstraint
from constraints.end_effector import EndEffectorConstraint
from constraints.kinetic_energy import KineticEnergyConstraint
from evaluator import get_all_metrics
from tqdm import tqdm
def main_root_control(opt):
    batch_size = opt.batch_size
    NUM_TIMESTEPS = opt.NUM_TIMESTEPS
    gt_motions_files = opt.gt_motions_files
    for index in tqdm(range(0, len(gt_motions_files), batch_size)):
        # Create a sublist of 100 elements
        sublist = gt_motions_files[index:index + batch_size]
        gt_samples = torch.stack([torch.load(os.path.join(opt.gt_motions_path, file)) for file in sublist])
        # add dummy contact
        if gt_samples.shape[2] == 135:
            gt_samples = torch.cat([torch.zeros(gt_samples.shape[0], gt_samples.shape[1], 4), gt_samples, ], dim=2)
        gt_samples_normalized = opt.model.normalizer.normalize(gt_samples)
        gt_root_normalized = gt_samples_normalized[:, :, 4:4+3] # first 4 are contact labels
        controlled_joint_indices = [0]
        const = TrajectoryConstraint3D(traj=gt_root_normalized, controlled_joint_indices=controlled_joint_indices)
        opt.constraint = const

        shape = gt_samples.shape
        extra_args = {}
        if opt.method == "dps":
            extra_args["weight"] = 0.1
            samples = model.diffusion.dps_sample(shape, sample_steps=NUM_TIMESTEPS, constraint_obj=opt.constraint,
                                                 weight=extra_args["weight"])
        elif opt.method == "dsg":
            extra_args["gr"] = 0.1
            samples = model.diffusion.dsg_sample(shape, sample_steps=NUM_TIMESTEPS, constraint_obj=opt.constraint,
                                                 gr=extra_args["gr"])
        elif opt.method == "trust":
            extra_args["norm_upper_bound"] = 80
            extra_args["iterations_max"] = 5
            extra_args["gradient_norm"] = 1
            extra_args["iteration_func"] = lambda time_next: 1  # 1
            model.diffusion.set_trust_parameters(iteration_func=extra_args["iteration_func"],
                                                 norm_upper_bound=extra_args["norm_upper_bound"],
                                                 iterations_max=extra_args["iterations_max"],
                                                 gradient_norm=extra_args["gradient_norm"])
            samples, traj_found = model.diffusion.trust_sample(shape, constraint_obj=opt.constraint, debug=True)
        if opt.save_motions:
            # make directory if it does not exist os.path.join(opt.motion_save_dir, f"{opt.method}")
            if not os.path.isdir(os.path.join(opt.motion_save_dir, f"{opt.method}")):
                os.makedirs(os.path.join(opt.motion_save_dir, f"{opt.method}"))
            for i in range(samples.shape[0]):
                sample = samples[i]
                samples_file = os.path.join(opt.motion_save_dir, f"{opt.method}", sublist[i])
                torch.save(sample, samples_file)

        if not opt.no_render:
            NUM_Render = min(2, samples.shape[0])
            print(f'Rendering some samples...')
            if not os.path.isdir(os.path.join(opt.render_dir, f"{opt.method}")):
                os.makedirs(os.path.join(opt.render_dir, f"{opt.method}"))
            for i in range(samples.shape[0]):
                render_dir = os.path.join(opt.render_dir, f"{opt.method}", sublist[i])
                sample = samples[i]
                sample_gt = gt_samples[i]
                render_samples = torch.stack([sample, sample_gt.to(sample.device)])
                render_to_compare(model.smpl, render_samples, model.normalizer, render_out=render_dir,
                                             constraint=None)


            # if opt.method == "trust":
            #     print(f'Rendering trajectory changes...')
            #     ani = trajectory_animation(traj_found, traj)
            #     ani.save(os.path.join(render_dir, 'trajectory.mp4'), writer='ffmpeg')

            print('Finished rendering samples.\n')


if __name__ == "__main__":
    opt = parse_test_opt()

    opt.save_motions = True
    opt.no_render = False
    opt.predict_contact = True
    opt.checkpoint = "../runs/motion/exp4-train-4950.pt"
    opt.model_name = "fixes_4950"
    opt.NUM_TIMESTEPS = 50
    opt.batch_size = 60
    print('**********************')
    print('Loading model...')
    model = MotionWrapper(opt.checkpoint, predict_contact=opt.predict_contact)
    model.eval()
    opt.model = model
    print('Model loaded.')
    print('**********************\n')


    opt.gt_motions_path = os.path.join(main_path, 'data', 'AMASS_test_aggregated_sliced')

    # get list of gt motions files
    opt.gt_motions_files = os.listdir(opt.gt_motions_path)


    # root control
    # We provide the 3D root trajectory during the full 3s as a constraint
    opt.motion_save_dir = "./motions/root_control"
    opt.render_dir = "renders/root_control"


    for method in ["dps", "dsg", "trust"][2:]:
        opt.method = method
        main_root_control(opt)