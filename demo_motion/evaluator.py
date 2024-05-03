import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import inspect
from einops import reduce
from model.motion_wrapper import MotionWrapper, SMPLSkeleton, ax_from_6v
from motion_args import parse_test_opt
from constraints.trajectory_constraint import TrajectoryConstraint
from constraints.specified_points import SpecifiedPointConstraint
from constraints.end_effector import EndEffectorConstraint
from constraints.kinetic_energy import KineticEnergyConstraint
def get_poses(samples: torch.FloatTensor) -> torch.FloatTensor:
    assert torch.cuda.is_available()
    smpl = SMPLSkeleton(device=torch.device('cuda'))

    b, s, c = samples.shape
    if c != 135: samples = samples[:, :, 4:]
    pos = samples[:, :, :3].to('cuda')
    q = samples[:, :, 3:].reshape(b, s, 22, 6)
    q = ax_from_6v(q).to('cuda') # go 6d to ax
    poses = smpl.forward(q, pos).detach().cpu().numpy()
    return poses


def get_foot_info(poses: np.ndarray, velo_heuristic=False) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    '''
    Assumes poses is of shape [batch size, # frames, 24, 3]
    '''
    # compute heuristic for feet contact
    feet = poses[:, :, (7, 8, 10, 11)]
    feetv = np.zeros(feet.shape[:3])
    feetv[:, :-1] = np.linalg.norm(feet[:, 1:] - feet[:, :-1], axis=-1)
    if velo_heuristic:
        HEURISTIC = 0.02  # in original code, was 0.01
        contact = feetv < HEURISTIC
    else:
        HEURISTIC = 0.03
        offset = np.array([0.95,0.95,1.00,1.00])
        offset = np.repeat(offset[np.newaxis, np.newaxis, :], feet.shape[1], axis=1)
        feetz = feet[..., 2] + offset  # feet elevation
        contact = feetz < HEURISTIC
    return torch.from_numpy(feetv), torch.from_numpy(contact)

def get_foot_loss(feetv: torch.FloatTensor, contact: torch.FloatTensor, loss_fn) -> torch.FloatTensor:
    # static_idx = contact > 0.95
    # feetv[~static_idx] = 0
    foot_loss = loss_fn(
        feetv, torch.zeros_like(feetv), reduction="none"
    )
    foot_loss = reduce(foot_loss, "b s ... -> b s", "mean")
    return foot_loss

def get_EDGE_PFC(samples: np.ndarray):
    '''
    Assumes samples is of shape [batch size, # frames, 139]
    '''
    DT = 1 / 30
    up_dir = 2
    flat_dirs = [i for i in range(3) if i != up_dir]

    poses = get_poses(samples)
    root_p = poses[...,0,:]
    root_v = (root_p[..., 1:, :] - root_p[..., :-1, :]) / DT  # root velocity (batch size, # frames-1, 3)
    root_a = (root_v[..., 1:, :] - root_v[..., :-1, :]) / DT  # (batch size, # frames-2, 3) root accelerations
    # clamp the up-direction of root acceleration
    root_a[..., up_dir] = np.maximum(root_a[..., up_dir], 0)  # (batch size, # frames-2, 3)
    # l2 norm
    root_a = np.linalg.norm(root_a, axis=-1)  # (batch size, # frames-2,)
    scaling = np.max(root_a,axis=1)
    scaling = np.repeat(scaling[:, np.newaxis], root_a.shape[1], axis=1)
    root_a /= scaling


    foot_idx = [7, 10, 8, 11]
    feet = poses[..., foot_idx, :]  # foot positions (batch size, # frames, 4, 3)
    foot_v = np.linalg.norm(
        feet[:, 2:, :, flat_dirs] - feet[:, 1:-1, :, flat_dirs], axis=-1
    )  # (batch_size, # frames-2, 4) horizontal velocity
    foot_mins = np.zeros((samples.shape[0], foot_v.shape[1], 2))
    foot_mins[..., 0] = np.minimum(foot_v[..., 0], foot_v[..., 1])
    foot_mins[..., 1] = np.minimum(foot_v[..., 2], foot_v[..., 3])
    PFC = (
            foot_mins[..., 0] * foot_mins[..., 1] * root_a
    )  # min leftv * min rightv * root_a (batch_size,# frames-2,)

    # Alternative way - only looking at slowest foot ~ one static foot can generate any acceleration
    # foot_mins = np.min(foot_mins, axis=-1)  # (batch_size, # frames-2, 2)
    # PFC = (
    #         np.square(foot_mins) * root_a
    # )  # min leftv * min rightv * root_a (batch_size,# frames-2,)

    PFC_mean = np.mean(PFC, axis=1) * 1e4 #scaling as in EDGE
    return PFC_mean

# def multivariate_gaussian_log_likelihood(sample, mean, covariance):
#     k = mean.shape[0]  # Dimensionality of the multivariate Gaussian
#     det = torch.det(covariance)
#     inv_covariance = torch.inverse(covariance)
#     exponent = -0.5 * torch.matmul(torch.matmul((sample - mean).T, inv_covariance), (sample - mean))
#     constant_term = -0.5 * k * torch.log(torch.tensor(2 * torch.pi))
#     log_likelihood = constant_term - 0.5 * torch.log(det) + exponent
#     return log_likelihood
def get_all_metrics(samples, constraint, model: MotionWrapper, exp_name=None, auto_save=True, file_path="all_motion_experiments.csv", **kwargs):
    if exp_name is None and auto_save: 
        raise ValueError("must have valid experiment name when autosaving")
    
    c_values = np.sqrt(np.abs(constraint.constraint(samples).cpu().numpy()))
    if np.isnan(c_values).any() or np.isinf(c_values).any():
        print('warning: constraint metric has nan or inf. cleaning this out')
        c_values = c_values[np.isfinite(c_values)]
    
    constraint_metrics = {
        "constraint mean": np.mean(c_values),
        # "constraint median": np.median(c_values),
        "constraint std": np.std(c_values)
    }
    
    ########## calculate distribution metrics ##
    # EDGE PFC metric
    PFC = get_EDGE_PFC(samples)
    print(PFC)

    poses = get_poses(samples)
    
    # mean foot loss (velocity)
    feetv, contact = get_foot_info(poses, velo_heuristic=True)
    mean_foot_loss_velo = torch.mean(get_foot_loss(feetv, contact, F.mse_loss), dim=-1)
    
    # mean foot loss (z)
    feetv, contact = get_foot_info(poses, velo_heuristic=False)
    mean_foot_loss_z = 1000*torch.mean(get_foot_loss(feetv, contact, F.mse_loss), dim=-1)
    
    # error predicted by model
    time_cond = torch.full((samples.shape[0],), 1, device=samples.device, dtype=torch.long)
    pred_noise, *_ = model.diffusion.model_predictions(samples, time_cond, clip_x_start=model.diffusion.clip_denoised)
    rms_pred_noise = torch.norm(pred_noise, dim=(-2,-1))
    # likelihood = multivariate_gaussian_log_likelihood(pred_noise[0,...].view(-1), 0*pred_noise[0,...].view(-1), torch.eye(pred_noise.shape[-1] * pred_noise.shape[-2], device=pred_noise.device))

    
    distr_metrics = {
        "foot_loss_z": torch.mean(mean_foot_loss_z).item(),
        "foot_loss_z_std": torch.std(mean_foot_loss_z).item(),
        "norm_pred_noise": torch.mean(rms_pred_noise).item(),
        "norm_pred_noise_std": torch.std(rms_pred_noise).item(),
        "PFC": np.mean(PFC).item(),
        "PFC_std": np.std(PFC).item(),
        "foot_loss_velo": torch.mean(mean_foot_loss_velo).item(),
        "foot_loss_velo_std": torch.std(mean_foot_loss_velo).item()
    }
    
    metrics = {
        "constraint": constraint_metrics,
        "distribution": distr_metrics
    }
    
    if auto_save:
        # convert metrics into better format
        better_metrics_dict = {}
        # add experiment name as a header
        better_metrics_dict["exp_name"] = exp_name
        for category in metrics:
            for metric in metrics[category]:
                better_metrics_dict[metric] = metrics[category][metric]
        for arg in kwargs:
            better_metrics_dict[arg] = kwargs[arg]
        _auto_save(better_metrics_dict, file_path=file_path)
    return metrics

def _auto_save(metrics, file_path="all_motion_experiments.csv"):
    if "iteration_func" in metrics:
        try:
            source = inspect.getsource(metrics['iteration_func'])
            metrics['iteration_func'] = source
        except IOError:
            print('could not extract function code')
    df_new = pd.DataFrame([metrics])

    if os.path.isfile(file_path):
        df_existing = pd.read_csv(file_path)
        df_final = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        print(f'File {file_path} does not exist, creating new file...')
        df_final = df_new

    df_final.to_csv(file_path, index=False)
    print(f"Data saved to {file_path} successfully.")





if __name__ == "__main__":
    opt = parse_test_opt()
    opt.motion_save_dir = "./motions"
    opt.render_dir = "renders/experimental"
    opt.save_motions = False
    opt.no_render = False
    opt.predict_contact = True
    opt.checkpoint = "../runs/motion/exp4-train-4950.pt"
    opt.model_name = "fixes_4950"
    opt.method = "dps"

    print('**********************')
    print('Loading model...')
    model = MotionWrapper(opt.checkpoint, predict_contact=opt.predict_contact)
    model.eval()
    print('Model loaded.')
    print('**********************\n')

    methods = ["dps", "dsg", "trust"]
    # Lshape
    for method in methods:
        opt.method = method
        X_START, Y_START = -0.107, -0.1545
        x_traj = torch.cat((torch.linspace(X_START, 0.2, 30), torch.linspace(0.2, 0.2, 30)))
        y_traj = torch.cat((torch.linspace(Y_START, Y_START, 30), torch.linspace(Y_START, 0.3, 30)))
        traj = torch.stack((x_traj, y_traj)).T
        const = TrajectoryConstraint(traj=traj)
        const.set_name("Lshape")
        opt.constraint = const

        samples = torch.load('./motions/' + opt.model_name + '_' + const.name + '/' + opt.method + '/normal_samples.pt')
        get_all_metrics(samples, opt.constraint, model, exp_name=f"{opt.model_name}_{opt.method}_{opt.constraint}")

    # specified_up_and_back
    for method in methods:
        opt.method = method
        points = [(0, 4, X_START), (30, 4, X_START), (59, 4, X_START),
                  (0, 5, Y_START), (30, 5, 0.3), (59, 5, Y_START)]
        const = SpecifiedPointConstraint(points=points)
        const.set_name("specified_up_and_back")
        opt.constraint = const

        samples = torch.load('./motions/' + opt.model_name + '_' + const.name + '/' + opt.method + '/normal_samples.pt')
        get_all_metrics(samples, opt.constraint, model, exp_name=f"{opt.model_name}_{opt.method}_{opt.constraint}")

    # specified_jump
    for method in methods:
        opt.method = method
        points = [(25, 6, 0.3), (30, 6, 0.8), (35, 6, 0.3)]
        const = SpecifiedPointConstraint(points=points)
        const.set_name("specified_jump")
        opt.constraint = const

        samples = torch.load('./motions/' + opt.model_name + '_' + const.name + '/' + opt.method + '/normal_samples.pt')
        get_all_metrics(samples, opt.constraint, model, exp_name=f"{opt.model_name}_{opt.method}_{opt.constraint}")

