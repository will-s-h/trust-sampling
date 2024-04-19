import os
import torch
import pandas as pd
import numpy as np
import inspect

def sanity_check(samples):
    samples = samples[-1]
    num_reasonable = sum((samples[:, 0] <= 2) & (samples[:, 0] >= -2) & (samples[:, 1] <= 2) & (samples[:, 1] >= -2))
    print(f'points within reasonable bounds: {num_reasonable}')

def get_all_metrics(samples, constraint, distr, exp_name=None, auto_save=True, file_path="all_2D_experiments.csv", **kwargs):
    if exp_name is None and auto_save: 
        raise ValueError("must have valid experiment name when autosaving")
    
    samples = samples[-1] # only consider last sample, should be shape [1000, 2] or similar
    c_values = np.abs(constraint.constraint(torch.from_numpy(samples).float()).numpy())
    if np.isnan(c_values).any() or np.isinf(c_values).any():
        print('warning: constraint metric has nan or inf. cleaning this out')
        c_values = c_values[np.isfinite(c_values)]
    
    constraint_metrics = {
        "mean": np.mean(c_values),
        "median": np.median(c_values),
        "std": np.std(c_values)
    }
    
    d_values = np.abs(np.array([distr.constraint(x).item() for x in torch.from_numpy(samples).float()]))
    if np.isnan(d_values).any() or np.isinf(d_values).any():
        print('warning: constraint metric has nan or inf. cleaning this out')
        d_values = d_values[np.isfinite(d_values)]
    
    distr_metrics = {
        "mean": np.mean(d_values),
        "median": np.median(d_values),
        "std": np.std(d_values)
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
                better_metrics_dict[category + " " + metric] = metrics[category][metric]
        for arg in kwargs:
            better_metrics_dict[arg] = kwargs[arg]
        _auto_save(better_metrics_dict, file_path=file_path)
    return metrics

def _auto_save(metrics, file_path="all_2D_experiments.csv"):
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