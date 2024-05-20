## to be able to import all modules
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

## other imports
import csv
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
from constraints.high_jump import HighJumpConstraint
from constraints.crawl import CrawlConstraint
from constraints.angular_momentum import AngularMomentumConstraint
from evaluator import get_all_metrics_
from tqdm import tqdm


def get_samples_NFEs(opt, shape):
    extra_args = {}
    NUM_TIMESTEPS = opt.NUM_TIMESTEPS
    NFEs = torch.ones(shape[0]) * NUM_TIMESTEPS
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
        extra_args["iterations_max"] = opt.J if hasattr(opt, 'J') and opt.J is not None else 5
        extra_args["gradient_norm"] = opt.gradient_norm if hasattr(opt, 'gradient_norm') and opt.gradient_norm is not None else 1
        extra_args["iteration_func"] = lambda time_next: 1  # 1
        opt.model.diffusion.set_trust_parameters(iteration_func=extra_args["iteration_func"],
                                                 norm_upper_bound=extra_args["norm_upper_bound"],
                                                 iterations_max=extra_args["iterations_max"],
                                                 gradient_norm=extra_args["gradient_norm"])
        samples, NFEs = model.diffusion.trust_sample(shape, sample_steps=NUM_TIMESTEPS, constraint_obj=opt.constraint, debug=True)
    return samples, NFEs


def main_angular_momentum(opt):
    shape = (1, 60, 139)
    targets = torch.tensor([5, 0])
    angular_momentum_constraint = AngularMomentumConstraint(0.8, targets)
    angular_momentum_constraint.set_smpl(opt.model.smpl)
    angular_momentum_constraint.set_normalizer(opt.model.normalizer)
    opt.constraint = angular_momentum_constraint

    samples, _ = get_samples_NFEs(opt, shape)
    long_sample = opt.constraint.stack_samples(samples)

    just_render_simple(
        opt.model.smpl,
        long_sample,
        opt.model.normalizer,
        opt.render_name + "_test")
    print()

    if opt.save_motions:
        # make directory if it does not exist os.path.join(opt.motion_save_dir, f"{opt.method}")
        if opt.method == "trust":
            refine_suffix = ''
            if opt.refine:
                refine_suffix = 'refine'
            motion_name = os.path.join(opt.motion_save_dir,
                                       f"{opt.method}" + str(opt.NUM_TIMESTEPS) + '_' + str(
                                           opt.max_norm) + '_' + refine_suffix)
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


def main_crawl(opt):
    shape = (1, 60, 139)
    targets = torch.tensor([5, 0])
    crawl_constraint = CrawlConstraint(0.7, targets)
    crawl_constraint.set_smpl(opt.model.smpl)
    crawl_constraint.set_normalizer(opt.model.normalizer)
    opt.constraint = crawl_constraint

    samples, _ = get_samples_NFEs(opt, shape)
    long_sample = opt.constraint.stack_samples(samples)

    just_render_simple(
        opt.model.smpl,
        long_sample,
        opt.model.normalizer,
        opt.render_name + "_test")
    print()

    if opt.save_motions:
        # make directory if it does not exist os.path.join(opt.motion_save_dir, f"{opt.method}")
        if opt.method == "trust":
            refine_suffix = ''
            if opt.refine:
                refine_suffix = 'refine'
            motion_name = os.path.join(opt.motion_save_dir,
                                       f"{opt.method}" + str(opt.NUM_TIMESTEPS) + '_' + str(
                                           opt.max_norm) + '_' + refine_suffix)
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


def main_high_jump(opt):
    batch_size = opt.batch_size
    NUM_TIMESTEPS = opt.NUM_TIMESTEPS
    heights = [0.6, 0.7, 0.8, 0.9, 1.0]
    constraint_violations = []
    NFEs = []
    pred_noises = []   
    for j in range(len(heights)):
        render_name_j = os.path.join(opt.render_name, "height_" + str(heights[j]))
        motion_name_j = os.path.join(opt.motion_name, "height_" + str(heights[j]))
        # make opt.render_name and opt.motion_name dirs if they do not exists
        if not os.path.isdir(render_name_j):
            os.makedirs(render_name_j)
        if not os.path.isdir(motion_name_j):
            os.makedirs(motion_name_j)
        shape = (50, 60, 139)
        targets = torch.tensor([5, 0])
        high_hand_constraint = HighJumpConstraint(heights[j], targets)
        high_hand_constraint.set_smpl(opt.model.smpl)
        high_hand_constraint.set_normalizer(opt.model.normalizer)
        opt.constraint = high_hand_constraint

        if opt.generate_motions:
            samples, sample_NFEs = get_samples_NFEs(opt, shape)
            just_render_simple(
                opt.model.smpl,
                samples[:5,...],
                opt.model.normalizer,
                render_name_j)

            if opt.save_motions:
                for i in range(samples.shape[0]):
                    sample = samples[i].clone()
                    sample_file = os.path.join(motion_name_j, str(i) + "_NFEs_" + str(sample_NFEs[i].item()))
                    torch.save(sample, sample_file)

        if opt.get_metrics:
            motion_list = os.listdir(motion_name_j)
            # load all motions
            samples = []
            sample_NFEs = []
            for motion_file in motion_list[:50]:
                samples.append(torch.load(os.path.join(motion_name_j, motion_file)))
                sample_NFEs.append(torch.tensor(float(motion_file.split('_')[-1])))
            samples = torch.stack(samples)
            # pred_noise = []
            # for b in range(samples.shape[0]):
            time_cond = torch.full((samples.shape[0],), int(20), device=samples.device, dtype=torch.long)
            pred_noise_, x_start, *_ = opt.model.diffusion.model_predictions(samples, time_cond, clip_x_start = True)
            pred_noise = torch.norm(pred_noise_.detach().cpu(), dim=(1,2))
            pred_noises.append(pred_noise/80)
            NFEs.append(torch.stack(sample_NFEs))
            constraint_violation = high_hand_constraint.vertical_location_violation(samples)
            constraint_violations.append(constraint_violation)
    if opt.get_metrics:
        NFEs = torch.stack(NFEs).flatten()
        constraint_violations = torch.stack(constraint_violations).flatten()
        NFEs_mean = torch.mean(NFEs)
        constraint_violations_mean = torch.mean(constraint_violations)
        NFEs_std = torch.std(NFEs)
        constraint_violations_std = torch.std(constraint_violations)
        pred_noises = torch.stack(pred_noises).flatten()
        pred_noises_mean = torch.mean(pred_noises)
        pred_noises_std = torch.std(pred_noises)

        return (NFEs_mean.item(), NFEs_std.item(), constraint_violations_mean.item(), constraint_violations_std.item(), pred_noises_mean.item(), pred_noises_std.item())


def main_obstacle_avoidance(opt):
    obstacles = torch.load('obstacles')
    targets = torch.load('targets')
    constraint_violations = []
    NFEs = []
    for j in range(targets.shape[0]):
        shape = (2, 60, 139)
        obstacles_j = obstacles[j,...].T
        # obstacle_list = [obstacles_j[:,i] for i in range(obstacles_j.shape[1])]
        obstacle_constraint = ObstacleAvoidance(obstacles_j, targets[j,:])
        obstacle_constraint.set_smpl(opt.model.smpl)
        obstacle_constraint.set_normalizer(opt.model.normalizer)
        opt.constraint = obstacle_constraint
        render_name_j = os.path.join(opt.render_name, str(j))
        motion_name_j = os.path.join(opt.motion_name, str(j))
        # make opt.render_name and opt.motion_name dirs if they do not exists
        if not os.path.isdir(render_name_j):
            os.makedirs(render_name_j)
        if not os.path.isdir(motion_name_j):
            os.makedirs(motion_name_j)

        if opt.generate_motions:
            samples, NFEs = get_samples_NFEs(opt, shape)
            long_sample = opt.constraint.stack_samples(samples)
            NFEs = torch.mean(NFEs)

            if j < 5:
                just_render_simple_wObstacles(
                    opt.model.smpl,
                    long_sample,
                    opt.model.normalizer,
                    render_name_j,
                    obstacles = obstacles_j)

            if opt.save_motions:
                sample_file = os.path.join(motion_name_j, str(j) + "_NFEs_" + str(NFEs.item()))
                torch.save(long_sample, sample_file)

        if opt.get_metrics:
            motion_list = os.listdir(motion_name_j)
            # load all motions
            samples = []
            sample_NFEs = []
            for motion_file in motion_list:
                samples.append(torch.load(os.path.join(motion_name_j, motion_file)))
                sample_NFEs.append(torch.tensor(float(motion_file.split('_')[-1])))
            samples = torch.stack(samples[:50])
            NFEs.append(torch.stack(sample_NFEs))
            constraint_violation = high_hand_constraint.vertical_location_violation(samples)
            constraint_violations.append(constraint_violation)


def main_obstacle_avoidance_(opt):
    obstacles = torch.load('obstacles')
    targets = torch.load('targets')
    shape = (2, 60, 139)

    obstacle_constraint = ObstacleAvoidance(obstacles, targets)
    obstacle_constraint.set_smpl(opt.model.smpl)
    obstacle_constraint.set_normalizer(opt.model.normalizer)
    opt.constraint = obstacle_constraint

    samples, _ = get_samples_NFEs(opt, shape)
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
    obstacles = torch.load('obstacles')
    targets = torch.load('targets')
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

    samples, _ = get_samples_NFEs(opt, shape)
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
    opt.control_name = "high_jump"
    opt.motion_save_dir = "./motions/" + opt.control_name
    opt.render_dir = "renders/" + opt.control_name
    opt.file_path = opt.control_name + ".csv"
    opt.auto_save = True
    opt.NUM_TIMESTEPS = 32
    max_norms = [80,82,84,1000] #80,81,82,
    Js = [3,4,5,6,7,8, 9, 10] #
    # Js = [8, 9, 10]

    opt.refine = False
    opt.generate_motions = True
    opt.get_metrics = True
    all_metrics = []
    metrics_cols = []
    metrics_rows = []
    for method in ["trust", "dsg", "dps"]:
        opt.method = method
        if opt.method == "trust":
            opt.NUM_TIMESTEPS = 50
            for i_max_norm in range(len(max_norms)):
                opt.max_norm = max_norms[i_max_norm]

                for i_J in range(len(Js)):

                    opt.J = Js[i_J]
                    opt.motion_name = os.path.join(opt.motion_save_dir,
                                                   f"{opt.method}" + str(opt.NUM_TIMESTEPS) + '_' + str(
                                                       opt.max_norm) + '_' + str(opt.J))
                    opt.render_name = os.path.join(opt.render_dir, f"{opt.method}" + str(opt.NUM_TIMESTEPS) + '_' + str(
                        opt.max_norm) + '_' + str(opt.J))
                    metrics = main_high_jump(opt)
                    all_metrics.append(metrics)
                    if opt.J < 10:
                        metrics_cols.append('0'+str(opt.J))
                    else:
                        metrics_cols.append(str(opt.J))
                    if opt.max_norm < 1000:
                        metrics_rows.append('00'+str(opt.max_norm))
                    else:
                        metrics_rows.append(str(opt.max_norm))
                    # main_obstacle_avoidance(opt)

        else:
            opt.NUM_TIMESTEPS = 200
            opt.motion_name = os.path.join(opt.motion_save_dir, f"{opt.method}" + str(opt.NUM_TIMESTEPS))
            opt.render_name = os.path.join(opt.render_dir, f"{opt.method}" + str(opt.NUM_TIMESTEPS))
            metrics = main_high_jump(opt)
            all_metrics.append(metrics)
            metrics_rows.append(f"{opt.method}")
            metrics_cols.append(f"{opt.method}")
            # main_obstacle_avoidance(opt)

    constraint_violations = [metric[2] for metric in all_metrics]
    constraint_violations_std = [metric[3] for metric in all_metrics]
    NFEs = [metric[0] for metric in all_metrics]
    NFEs_std = [metric[1] for metric in all_metrics]
    pred_noises = [metric[4] for metric in all_metrics]
    pred_noises_std = [metric[5] for metric in all_metrics]
    all_metrics = [NFEs, constraint_violations, pred_noises]
    score_names = ['NFEs', 'constraint_violations', 'pred_noises']
    # generate a csv file with metrics and colheaders for rows and cols are described by metrics_rows and metrics_cols
    output_file = 'high_jump_constraint_violations.csv'
    # Write to CSV


    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        for metrics_i, metrics in enumerate(all_metrics):
            #open new sheet in csv
            writer.writerow([''])
            writer.writerow([''])
            writer.writerow([score_names[metrics_i]])


            # Write the column headers
            col_headers = list(set(metrics_cols))
            col_headers.sort()
            row_headers = list(set(metrics_rows))
            row_headers.sort()
            writer.writerow([''] + col_headers)

            # Write the data rows
            for row_header in row_headers:
                score_row = []
                for col_header in col_headers:
                    # get allindices in metric_rows that correspond to row_header
                    row_indices = [i for i in range(len(metrics_rows)) if metrics_rows[i] == row_header]
                    col_indices = [i for i in range(len(metrics_cols)) if metrics_cols[i] == col_header]
                    # find the intersection of row_indices and col_indices
                    indices = list(set(row_indices).intersection(col_indices))
                    if indices != []:
                        score_row.append(metrics[indices[0]])
                    else:
                        score_row.append('')
                writer.writerow([row_header] + list(score_row))

    print('Metrics saved to:', output_file)