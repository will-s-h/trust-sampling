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
from visualization.render_motion import just_render_simple
from constraints.trajectory_constraint_3d import TrajectoryConstraint3D
from constraints.end_effector import EndEffectorConstraintFootHand
from constraints.long_form_motion import LongFormMotion
from evaluator import get_all_metrics_
from tqdm import tqdm

def main_long_form_control(opt):
    batch_size = opt.batch_size
    NUM_TIMESTEPS = opt.NUM_TIMESTEPS
    gt_motions_files = opt.gt_motions_files
    shape = (5, 60, 139)
    const = LongFormMotion(shape[0])
    const.set_normalizer(opt.model.normalizer)
    opt.constraint = const

    extra_args = {}
    if opt.method == "dps":
        extra_args["weight"] = 0.1
        samples = opt.model.diffusion.dps_sample(shape, sample_steps=NUM_TIMESTEPS, constraint_obj=opt.constraint,
                                                 weight=extra_args["weight"])
    elif opt.method == "dsg":
        extra_args["gr"] = 0.1
        samples = opt.model.diffusion.dsg_sample(shape, sample_steps=NUM_TIMESTEPS, constraint_obj=opt.constraint,
                                                 gr=extra_args["gr"])
    elif opt.method == "trust":
        extra_args["norm_upper_bound"] = opt.max_norm
        extra_args["iterations_max"] = 5
        extra_args["gradient_norm"] = 1
        extra_args["iteration_func"] = lambda time_next: 1  # 1
        opt.model.diffusion.set_trust_parameters(iteration_func=extra_args["iteration_func"],
                                                 norm_upper_bound=extra_args["norm_upper_bound"],
                                                 iterations_max=extra_args["iterations_max"],
                                                 gradient_norm=extra_args["gradient_norm"])
        samples, traj_found = model.diffusion.trust_sample(shape, sample_steps=NUM_TIMESTEPS,
                                                           constraint_obj=opt.constraint, debug=True)

        long_sample = const.stack_samples(samples)
        render_name = os.path.join(opt.render_dir, f"{opt.method}" + str(opt.NUM_TIMESTEPS)+ '_' +str(opt.max_norm))
        just_render_simple(
            opt.model.smpl,
        long_sample,
        opt.model.normalizer,
        render_name)
        print()


def main_end_effector_control(opt):
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
        const = EndEffectorConstraintFootHand(gt_samples_normalized)
        const.set_normalizer(opt.model.normalizer)
        const.set_smpl(opt.model.smpl)
        const.set_targets()
        opt.constraint = const

        shape = gt_samples.shape
        # shape_ddim = (1,160,139)
        # model.diffusion.ddim_sample(shape_ddim, sample_steps=NUM_TIMESTEPS)
        extra_args = {}
        if opt.method == "dps":
            extra_args["weight"] = 0.1
            samples = opt.model.diffusion.dps_sample(shape, sample_steps=NUM_TIMESTEPS, constraint_obj=opt.constraint,
                                                 weight=extra_args["weight"])
        elif opt.method == "dsg":
            extra_args["gr"] = 0.1
            samples = opt.model.diffusion.dsg_sample(shape, sample_steps=NUM_TIMESTEPS, constraint_obj=opt.constraint,
                                                 gr=extra_args["gr"])
        elif opt.method == "trust":
            extra_args["norm_upper_bound"] = opt.max_norm
            extra_args["iterations_max"] = 5
            extra_args["gradient_norm"] = 1
            extra_args["iteration_func"] = lambda time_next: 1  # 1
            opt.model.diffusion.set_trust_parameters(iteration_func=extra_args["iteration_func"],
                                                 norm_upper_bound=extra_args["norm_upper_bound"],
                                                 iterations_max=extra_args["iterations_max"],
                                                 gradient_norm=extra_args["gradient_norm"])
            samples, traj_found = model.diffusion.trust_sample(shape, sample_steps=NUM_TIMESTEPS, constraint_obj=opt.constraint, debug=True)
        if opt.save_motions:
            # make directory if it does not exist os.path.join(opt.motion_save_dir, f"{opt.method}")
            if opt.method == "trust":
                motion_name = os.path.join(opt.motion_save_dir, f"{opt.method}" + str(opt.NUM_TIMESTEPS)+ '_' +str(opt.max_norm))
            else:
                motion_name = os.path.join(opt.motion_save_dir, f"{opt.method}" + str(opt.NUM_TIMESTEPS))
            if not os.path.isdir(motion_name):
                os.makedirs(motion_name)
            for i in range(samples.shape[0]):
                sample = samples[i]
                samples_file = os.path.join(motion_name, sublist[i])
                torch.save(sample.clone(), samples_file)

        if not opt.no_render:
            NUM_Render = min(2, samples.shape[0])
            print(f'Rendering some samples...')
            if opt.method == "trust":
                render_name = os.path.join(opt.render_dir, f"{opt.method}" + str(opt.NUM_TIMESTEPS)+ '_' +str(opt.max_norm))
            else:
                render_name = os.path.join(opt.render_dir, f"{opt.method}" + str(opt.NUM_TIMESTEPS))
            if not os.path.isdir(render_name):
                os.makedirs(render_name)
            for i in range(NUM_Render):
                render_dir = os.path.join(render_name, sublist[i])
                sample = samples[i]
                sample_gt = gt_samples[i]
                render_samples = torch.stack([sample, sample_gt.to(sample.device)])
                render_to_compare(model.smpl, render_samples, model.normalizer, render_out=render_dir,
                                             constraint=None)

            if opt.get_metrics:
                get_all_metrics_(opt)
            # if opt.method == "trust":
            #     print(f'Rendering trajectory changes...')
            #     ani = trajectory_animation(traj_found, traj)
            #     ani.save(os.path.join(render_dir, 'trajectory.mp4'), writer='ffmpeg')

            print('Finished rendering samples.\n')


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
        # shape_ddim = (1,160,139)
        # model.diffusion.ddim_sample(shape_ddim, sample_steps=NUM_TIMESTEPS)
        extra_args = {}
        if opt.method == "dps":
            extra_args["weight"] = 0.1
            samples = opt.model.diffusion.dps_sample(shape, sample_steps=NUM_TIMESTEPS, constraint_obj=opt.constraint,
                                                 weight=extra_args["weight"])
        elif opt.method == "dsg":
            extra_args["gr"] = 0.1
            samples = opt.model.diffusion.dsg_sample(shape, sample_steps=NUM_TIMESTEPS, constraint_obj=opt.constraint,
                                                 gr=extra_args["gr"])
        elif opt.method == "trust":
            extra_args["norm_upper_bound"] = opt.max_norm
            extra_args["iterations_max"] = 5
            extra_args["gradient_norm"] = 1
            extra_args["iteration_func"] = lambda time_next: 1  # 1
            opt.model.diffusion.set_trust_parameters(iteration_func=extra_args["iteration_func"],
                                                 norm_upper_bound=extra_args["norm_upper_bound"],
                                                 iterations_max=extra_args["iterations_max"],
                                                 gradient_norm=extra_args["gradient_norm"])
            samples, traj_found = model.diffusion.trust_sample(shape, sample_steps=NUM_TIMESTEPS, constraint_obj=opt.constraint, debug=True)
        if opt.save_motions:
            # make directory if it does not exist os.path.join(opt.motion_save_dir, f"{opt.method}")
            if opt.method == "trust":
                motion_name = os.path.join(opt.motion_save_dir, f"{opt.method}" + str(opt.NUM_TIMESTEPS)+ '_' +str(opt.max_norm))
            else:
                motion_name = os.path.join(opt.motion_save_dir, f"{opt.method}" + str(opt.NUM_TIMESTEPS))
            if not os.path.isdir(motion_name):
                os.makedirs(motion_name)
            for i in range(samples.shape[0]):
                sample = samples[i]
                samples_file = os.path.join(motion_name, sublist[i])
                torch.save(sample.clone(), samples_file)

        if not opt.no_render:
            NUM_Render = min(2, samples.shape[0])
            print(f'Rendering some samples...')
            if opt.method == "trust":
                render_name = os.path.join(opt.render_dir, f"{opt.method}" + str(opt.NUM_TIMESTEPS)+ '_' +str(opt.max_norm))
            else:
                render_name = os.path.join(opt.render_dir, f"{opt.method}" + str(opt.NUM_TIMESTEPS))
            if not os.path.isdir(render_name):
                os.makedirs(render_name)
            for i in range(NUM_Render):
                render_dir = os.path.join(render_name, sublist[i])
                sample = samples[i]
                sample_gt = gt_samples[i]
                render_samples = torch.stack([sample, sample_gt.to(sample.device)])
                render_to_compare(model.smpl, render_samples, model.normalizer, render_out=render_dir,
                                             constraint=None)
        if opt.get_metrics:
            get_all_metrics_(opt)

            # if opt.method == "trust":
            #     print(f'Rendering trajectory changes...')
            #     ani = trajectory_animation(traj_found, traj)
            #     ani.save(os.path.join(render_dir, 'trajectory.mp4'), writer='ffmpeg')

            print('Finished rendering samples.\n')


if __name__ == "__main__":


    # GENERAL SETTINGS
    opt = parse_test_opt()
    opt.save_motions = True
    opt.no_render = False
    opt.predict_contact = True
    opt.checkpoint = "../runs/motion/exp4-train-4950.pt"
    opt.model_name = "fixes_4950"
    opt.batch_size = 60
    print('**********************')
    print('Loading model...')
    model = MotionWrapper(opt.checkpoint, predict_contact=opt.predict_contact)
    model.eval()
    opt.model = model
    print('Model loaded.')
    print('**********************\n')

    # GT MOTIONS
    opt.nr_test_motions = 480
    opt.gt_motions_path = os.path.join(main_path, 'data', 'AMASS_test_aggregated_sliced')
    opt.gt_motions_files = os.listdir(opt.gt_motions_path)[:opt.nr_test_motions]

    # TEST SETTINGS
    opt.control_name = "root_control" # or "end_effector_foot_hand"
    opt.get_metrics = False
    NUM_TIMESTEPS = [50, 200, 1000] # 1000 will be ignored for trust sampling
    max_norms = [[80, 82], [88, 88.5]] # provide different max norms per NUM_TIMESTEPS you want to try
    opt.max_norm_original = 80 # Need to specify for fair comparison in metrics

    # SAVE SETTINGS
    opt.motion_save_dir = "./motions/" + opt.control_name
    opt.render_dir = "renders/" + opt.control_name
    opt.file_path = opt.control_name + ".csv"
    opt.auto_save = True

    for i, NUM_TIMESTEP in enumerate(NUM_TIMESTEPS):
        opt.NUM_TIMESTEPS = NUM_TIMESTEP
        for method in ["dps", "dsg", "trust"][1:]:
            if method == 'trust' and NUM_TIMESTEP < 201:
                for max_norm in max_norms[i]:
                    opt.max_norm = max_norm
                    opt.method = method
                    opt.generated_motions_path = os.path.join(opt.motion_save_dir,
                                                              f"{opt.method}" + str(opt.NUM_TIMESTEPS) + '_' + str(
                                                                  opt.max_norm))
                    opt.generated_motion_files = os.listdir(os.path.join(opt.motion_save_dir, f"{opt.method}" + str(
                        opt.NUM_TIMESTEPS) + '_' + str(opt.max_norm)))
                    if opt.control_name == "root_control":
                        main_root_control(opt)
                    elif opt.control_name == "end_effector_foot_hand":
                        main_end_effector_control(opt)
            elif method != "trust":
                opt.method = method
                opt.generated_motions_path = os.path.join(opt.motion_save_dir, f"{opt.method}" + str(opt.NUM_TIMESTEPS))
                opt.generated_motion_files = os.listdir(
                    os.path.join(opt.motion_save_dir, f"{opt.method}" + str(opt.NUM_TIMESTEPS)))
                if opt.control_name == "root_control":
                    main_root_control(opt)
                elif opt.control_name == "end_effector_foot_hand":
                    main_end_effector_control(opt)



