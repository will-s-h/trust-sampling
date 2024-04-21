import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import inspect
from einops import reduce
from model.motion_wrapper import MotionWrapper, SMPLSkeleton, ax_from_6v

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
        HEURISTIC = 0.05
        feetz = feet[..., 2]  # feet elevation
        contact = feetz < HEURISTIC
    return torch.from_numpy(feetv), torch.from_numpy(contact)

def get_foot_loss(feetv: torch.FloatTensor, contact: torch.FloatTensor, loss_fn) -> torch.FloatTensor:
    static_idx = contact > 0.95
    feetv[~static_idx] = 0
    foot_loss = loss_fn(
        feetv, torch.zeros_like(feetv), reduction="none"
    )
    foot_loss = reduce(foot_loss, "b s ... -> b s", "mean")
    return foot_loss

def get_all_metrics(samples, constraint, model: MotionWrapper, exp_name=None, auto_save=True, file_path="all_motion_experiments.csv", **kwargs):
    if exp_name is None and auto_save: 
        raise ValueError("must have valid experiment name when autosaving")
    
    c_values = np.abs(constraint.constraint(samples).cpu().numpy())
    if np.isnan(c_values).any() or np.isinf(c_values).any():
        print('warning: constraint metric has nan or inf. cleaning this out')
        c_values = c_values[np.isfinite(c_values)]
    
    constraint_metrics = {
        "constraint mean": np.mean(c_values),
        "constraint median": np.median(c_values),
        "constraint std": np.std(c_values)
    }
    
    ########## calculate distribution metrics ##
    poses = get_poses(samples)
    
    # mean foot loss (velocity)
    feetv, contact = get_foot_info(poses, velo_heuristic=True)
    mean_foot_loss_velo = torch.mean(get_foot_loss(feetv, contact, F.mse_loss)).item()
    
    # mean foot loss (z)
    feetv, contact = get_foot_info(poses, velo_heuristic=False)
    mean_foot_loss_z = torch.mean(get_foot_loss(feetv, contact, F.mse_loss)).item()
    
    # error predicted by model
    time_cond = torch.full((samples.shape[0],), 0, device=samples.device, dtype=torch.long)
    pred_noise, *_ = model.diffusion.model_predictions(samples, time_cond, clip_x_start=model.diffusion.clip_denoised)
    rms_pred_noise = torch.norm(pred_noise).item()
    
    distr_metrics = {
        "mean_foot_loss_velo": mean_foot_loss_velo,
        "mean_foot_loss_z": mean_foot_loss_z,
        "rms_pred_noise": rms_pred_noise
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