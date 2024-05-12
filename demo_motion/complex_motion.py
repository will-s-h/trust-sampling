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
from visualization.render_motion import just_render_simple, just_render_simple_wObstacles
from constraints.trajectory_constraint_3d import TrajectoryConstraint3D
from constraints.end_effector import EndEffectorConstraintFootHand
from constraints.long_form_motion import LongFormMotion
from constraints.obstacle import ObstacleAvoidance
from evaluator import get_all_metrics_
from tqdm import tqdm



def main_obstacle_avoidance(opt):
    batch_size = opt.batch_size
    NUM_TIMESTEPS = opt.NUM_TIMESTEPS
    shape = (2, 60, 139)

    obstacles = [torch.tensor([1, 0, 0, 0.50])]
    targets = torch.tensor([5, 0])
    obstacle_constraint = ObstacleAvoidance(obstacles, targets)
    obstacle_constraint.set_smpl(opt.model.smpl)
    obstacle_constraint.set_normalizer(opt.model.normalizer)

    opt.constraint = obstacle_constraint

    # sample = opt.model.diffusion.ddim_sample(shape)
    # sample = torch.ones((shape)).to('cuda')
    # obstacle_constraint.constraint(sample)

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
                                                 gradient_norm=extra_args["gradient_norm"],
                                                 refine=opt.refine
                                                 )
        samples, traj_found = model.diffusion.trust_sample(shape, sample_steps=NUM_TIMESTEPS,
                                                           constraint_obj=opt.constraint, debug=True)

    long_sample = opt.constraint.stack_samples(samples)

    just_render_simple_wObstacles(
        opt.model.smpl,
    long_sample,
    opt.model.normalizer,
    opt.render_name + "_test",
    obstacles=obstacles)
    print()

    if opt.save_motions:
        # make directory if it does not exist os.path.join(opt.motion_save_dir, f"{opt.method}")
        if opt.method == "trust":
            refine_suffix = ''
            if opt.refine:
                refine_suffix = 'refine'
            motion_name = os.path.join(opt.motion_save_dir,
                                       f"{opt.method}" + str(opt.NUM_TIMESTEPS) + '_' + str(opt.max_norm) + '_' +refine_suffix)
        else:
            motion_name = os.path.join(opt.motion_save_dir, f"{opt.method}" + str(opt.NUM_TIMESTEPS))
        if not os.path.isdir(motion_name):
            os.makedirs(motion_name)
        # for i in range(samples.shape[0]):
        #     sample = samples[i]
        samples_file = os.path.join(motion_name, "_test")
        # if there is a nan value in samples print
        if torch.isnan(samples).any():
            print()
        torch.save(samples, samples_file)



def main_long_form_control(opt):
    batch_size = opt.batch_size
    NUM_TIMESTEPS = opt.NUM_TIMESTEPS
    shape = (4, 60, 139)
    const = LongFormMotion(shape[0], opt.target)
    const.set_normalizer(opt.model.normalizer)
    opt.constraint = const


    # generate two ddim samples
    sample = opt.model.diffusion.ddim_sample(shape)
    obstacles = [torch.tensor([1, 0, 0, 1])]
    targets = torch.tensor([3, 0])
    obstacle_constraint = ObstacleAvoidance(obstacles, targets)
    obstacle_constraint.set_smpl(opt.model.smpl)
    obstacle_constraint.set_normalizer(opt.model.normalizer)
    # sample = torch.ones((shape)).to('cuda')
    obstacle_constraint.constraint(sample)

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
                                                 gradient_norm=extra_args["gradient_norm"],
                                                 refine=opt.refine
                                                 )
        samples, traj_found = model.diffusion.trust_sample(shape, sample_steps=NUM_TIMESTEPS,
                                                           constraint_obj=opt.constraint, debug=True)

    long_sample = const.stack_samples(samples)

    just_render_simple(
        opt.model.smpl,
    long_sample,
    opt.model.normalizer,
    opt.render_name + "_x"+ str(const.x_target.item()) + "_y"+ str(const.y_target.item()))
    print()

    if opt.save_motions:
        # make directory if it does not exist os.path.join(opt.motion_save_dir, f"{opt.method}")
        if opt.method == "trust":
            refine_suffix = ''
            if opt.refine:
                refine_suffix = 'refine'
            motion_name = os.path.join(opt.motion_save_dir,
                                       f"{opt.method}" + str(opt.NUM_TIMESTEPS) + '_' + str(opt.max_norm) + '_' +refine_suffix)
        else:
            motion_name = os.path.join(opt.motion_save_dir, f"{opt.method}" + str(opt.NUM_TIMESTEPS))
        if not os.path.isdir(motion_name):
            os.makedirs(motion_name)
        # for i in range(samples.shape[0]):
        #     sample = samples[i]
        samples_file = os.path.join(motion_name, "_x"+ str(const.x_target.item()) + "_y"+ str(const.y_target.item()))
        print("_x"+ str(const.x_target.item()) + "_y"+ str(const.y_target.item()))
        # if there is a nan value in samples print
        if torch.isnan(samples).any():
            print()
        torch.save(samples, samples_file)

def get_metrics(opt, motion_list):
    # if opt.method == "trust":
    #     motion_name = os.path.join(opt.motion_save_dir,
    #                                f"{opt.method}" + str(opt.NUM_TIMESTEPS) + '_' + str(opt.max_norm))
    # else:
    #     motion_name = os.path.join(opt.motion_save_dir, f"{opt.method}" + str(opt.NUM_TIMESTEPS))
    motion_name = opt.motion_name
    distances = []
    continuities = []
    for motion_file in motion_list:
        motion = torch.load(os.path.join(motion_name, motion_file))
        target_x = float(motion_file.split('x')[1].split('_')[0])
        target_y = float(motion_file.split('y')[1])
        target = torch.tensor([target_x, target_y])
        shape = motion.shape
        const = LongFormMotion(shape[0], target)
        const.set_normalizer(opt.model.normalizer)
        long_sample = const.normalizer.unnormalize(const.stack_samples(motion)).squeeze()
        x_end = long_sample[-1, 4]
        y_end = long_sample[-1, 5]
        distance_to_cover = torch.sqrt(torch.tensor(target_x**2 + target_y**2))
        # if distance_to_cover < torch.sqrt(torch.tensor(32)):
        #     continue
        distance = torch.sqrt((x_end - target_x)**2 + (y_end - target_y)**2)
        distances.append(distance)
        continuity = torch.sqrt(torch.abs(const.constraint(motion.detach())[0]))
        continuities.append(continuity)
        if torch.isnan(distance):
            print()
    print('number of samples:', len(distances))
    print('Distance:', torch.nanmean(torch.stack(distances)))
    # print 95 percentile
    print('95 percentile:', torch.kthvalue(torch.stack(distances), int(0.95 * len(distances)))[0])
    print('Continuity:', torch.nanmean(torch.stack(continuities)))
    # print 95 percentile
    print('95 percentile:', torch.kthvalue(torch.stack(continuities), int(0.95 * len(continuities)))[0])




if __name__ == "__main__":

    # x_target = 8 * (torch.rand((500,)) )
    # y_target = 8 * (torch.rand((500,)) )
    # targets = torch.stack((x_target, y_target), dim=1)
    # torch.save(targets, "targets_long_form.pt")
    targets_long_form = torch.load("targets_long_form.pt")

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


    # SAVE SETTINGS
    opt.control_name = "obstacle_avoidance"
    opt.motion_save_dir = "./motions/" + opt.control_name
    opt.render_dir = "renders/" + opt.control_name
    opt.file_path = opt.control_name + ".csv"
    opt.auto_save = True
    opt.NUM_TIMESTEPS = 32
    opt.max_norm = 80
    opt.refine = True

    # observe sample
    # file_path = os.path.join(opt.motion_save_dir, "trust50_80_refine", "_test")
    # motion = torch.load(file_path)
    # obstacle_constraint = ObstacleAvoidance([torch.tensor([1, 0, 0, 0.25])], torch.tensor([3, 0]))
    # obstacle_constraint.set_normalizer(opt.model.normalizer)
    # # obstacle_constraint.constraint(motion)
    #
    # motion_pt1 = motion[0].cpu().detach().numpy()
    # motion_pt2 = motion[1].cpu().detach().numpy()
    for method in ["trust", "dsg", "dps"][1:]:
        opt.method = method
        if method != "trust":
            opt.NUM_TIMESTEPS = 200
        else:
            opt.NUM_TIMESTEPS = 50
        for i in range(100):
            opt.render_name = os.path.join(opt.render_dir, f"{opt.method}" + str(opt.NUM_TIMESTEPS)+ '_' + str(opt.max_norm), str(i))
            opt.target = targets_long_form[i]
            main_obstacle_avoidance(opt)

        if opt.method == "trust":
            refine_suffix = ''
            if opt.refine:
                refine_suffix = 'refine'
            motion_name = os.path.join(opt.motion_save_dir,
                                       f"{opt.method}" + str(opt.NUM_TIMESTEPS) + '_' + str(opt.max_norm) + '_' + refine_suffix)
        else:
            motion_name = os.path.join(opt.motion_save_dir, f"{opt.method}" + str(opt.NUM_TIMESTEPS))
        opt.motion_name = motion_name
        motion_list = os.listdir(motion_name)
        get_metrics(opt, motion_list)


