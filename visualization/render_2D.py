import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from dataset.datasets_2D import spiral_dataset, dino_dataset, ring_dataset
# from .ddpm import MLP

def _plot_distribution(ax=None, distr="spiral"):
    fig = None
    if ax is None: fig, ax = plt.subplots()
    if distr == "spiral":
        truth = spiral_dataset(n=200, ordered=True)
        ax.plot(truth[:, 0], truth[:, 1])
        return fig, ax
    elif distr == "ring":
        circle1 = Circle((0, 0), 1, fill=False, color='blue')
        circle2 = Circle((0, 0), 2, fill=False, color='blue')
        ax.add_patch(circle1)
        ax.add_patch(circle2)
        return fig, ax
    else:
        raise NotImplementedError(f"Have not implemented distribution plotting for {distr}")

def _plot_constraint(ax=None, constraint=None):
    fig = None
    if ax is None: fig, ax = plt.subplots()
    if constraint is None: return fig, ax
    try:  # should work for any constraint where gety is defined
        xs = np.linspace(-2, 2, 200)
        ys = constraint.gety(xs)
        xs = xs[(-2 <= ys) & (ys <= 2)]
        ys = ys[(-2 <= ys) & (ys <= 2)]
        ax.plot(xs, ys, alpha=0.5)
        return fig, ax
    except:
        raise NotImplementedError(f"No implementation of plotting constraint of type {type(constraint)}")

def plot_constraint_exp(samples, constraint, distr="spiral"):
    '''
    Plots the final results of constrained diffusion inference.
    '''
    fig, ax = _plot_distribution(distr=distr)
    _plot_constraint(ax, constraint)
    for idx in range(len(samples[0])):
        xs = [sample[idx, 0] for sample in samples[:]]
        ys = [sample[idx, 1] for sample in samples[:]]
        ax.scatter(xs[-1], ys[-1], s=15, alpha=0.2)
    ax.set_title("Constraint Experiment")
    ax.set_ylabel("y value")
    ax.set_xlabel("x value")
    return fig, ax

def video_all_steps(samples, constraint=None, distr="spiral"):
    fig, ax = _plot_distribution(distr=distr)
    _plot_constraint(ax, constraint)
    scatter = ax.scatter([], [], s=15, alpha=0.2)
    frame_text = ax.text(0.02, 1.03, '', transform=ax.transAxes)
    
    def update(frame):
        cur_frame = samples[frame]
        xs = [cur_frame[idx, 0] for idx in range(len(cur_frame))]
        ys = [cur_frame[idx, 1] for idx in range(len(cur_frame))]
        scatter.set_offsets(np.column_stack([xs, ys]))
        frame_text.set_text(f'Diffusion Timestep: {frame}')
        return scatter, frame_text
    
    ani = animation.FuncAnimation(fig, update, frames=len(samples), blit=True)
    return ani

def plot_gradient_vectors(constraint=None, distr="spiral"):
    assert constraint is not None
    fig, ax = _plot_distribution(distr=distr)
    _plot_constraint(ax, constraint)
    
    feature_x = np.linspace(-2, 2, 30)
    feature_y = np.linspace(-2, 2, 30)
    x, y = np.meshgrid(feature_x, feature_y)
    samples = torch.from_numpy(np.stack((x.flatten(), y.flatten())).T)
    gradients = -constraint.gradient(samples).numpy()
    
    u, v = gradients[:, 0], gradients[:, 1]
    norm = np.linalg.norm(gradients, axis=1)
    u /= norm
    v /= norm
    ax.quiver(x, y, u, v, scale=50)
    
    return fig, ax

# def plot_heat_map(model_or_constraint, distr="spiral"):
#     fig, ax = plt.subplots()#_plot_distribution(distr=distr)
    
#     N, M, NUM_CONTOURS = 500, 500, 20
#     feature_x = np.linspace(-2, 2, N)
#     feature_y = np.linspace(-2, 2, M)
#     x, y = np.meshgrid(feature_x, feature_y)
#     samples = torch.from_numpy(np.stack((x.flatten(), y.flatten())).T).float()
    
#     if isinstance(model_or_constraint, MLP):
#         t = torch.tensor(0).repeat(samples.shape[0]).long()
#         with torch.no_grad():
#             residual = model_or_constraint(samples, t).numpy()
#         vals = np.linalg.norm(residual, axis=1).reshape((N, M))
#     else: # assume it is a constraint
#         vals = torch.abs(model_or_constraint.constraint(samples)).numpy().reshape((N, M))
    
#     contour = ax.contourf(x, y, vals, NUM_CONTOURS, cmap='hot')
#     fig.colorbar(contour, ax=ax)
#     return fig, ax
    

def plot_flow_model(flow_model, distr="spiral"):
    fig, ax = plt.subplots()
    if distr == "spiral": 
        truth = torch.from_numpy(spiral_dataset(n=200, ordered=True)).float()
    elif distr == "dino":
        truth = torch.from_numpy(dino_dataset(n=200, numpy=True)).float()
    elif distr == 'ring':
        truth = torch.from_numpy(ring_dataset(n=200, numpy=True)).float()
    else:
        raise NotImplementedError("Non spiral distribution hasn't been implemented for plot_flow_model()")
    with torch.no_grad():
        vectors = flow_model(truth)
    ax.quiver(truth[:, 0], truth[:, 1], vectors[:, 0], vectors[:, 1], scale=40)
    return fig, ax
    

def plot_diffusion_steps(samples, distribution, cutoff=0.05):
    '''
    Plots how x changes after every diffusion timestep.
    '''
    fig, ax = plt.subplots()
    for idx in range(len(samples)):
        xs = [sample[idx, 0] for sample in samples]
        ys = [sample[idx, 1] for sample in samples]
        alpha=0.2
        if distribution.constraint((xs[-1], ys[-1])) > cutoff:
            alpha=0.2
        base_line, = ax.plot(xs, np.linspace(0,len(xs), len(xs)), alpha=alpha, marker='.', markersize=1)
        ax.scatter(xs, np.linspace(0,len(xs), len(xs)), s=1, alpha=alpha, color=base_line.get_color())
    ax.set_title('Diffusion Steps Visualized')
    ax.set_ylabel("diffusion timestep")
    ax.set_xlabel("x value")
    return fig, ax

def visualize_path(samples, num_points=50, skip_steps=1, distr="spiral"):
    '''
    Plots paths of several unconstrained points.
    '''
    fig, ax = _plot_distribution(distr=distr)
    idxs = random.sample(range(len(samples)), num_points)
    for idx in idxs:
        xs = [sample[idx, 0] for sample in samples]
        ys = [sample[idx, 1] for sample in samples]
        base_line, = ax.plot(xs[::skip_steps], ys[::skip_steps], alpha=0.5, marker='.', markersize=1)
        ax.scatter(xs[-1], ys[-1], s=15, color=base_line.get_color())
    ax.set_title('2D Path Visualization')
    ax.set_ylabel('y value')
    ax.set_xlabel('x value')
    return fig, ax

##### evaluation of how good it is

def distribution_metric(samples, model, return_metrics=False):
    samples = samples[-1] # only consider last sample, should be shape [1000, 2] or similar
    samples = torch.from_numpy(samples).float()
    t = torch.tensor(0).repeat(samples.shape[0]).long()
    with torch.no_grad():
        residual = model(samples, t).numpy()
    residual = np.linalg.norm(residual, axis=-1)
    if np.isnan(residual).any() or np.isinf(residual).any():
        print('warning: distribution metric has nan or inf. cleaning this out')
        residual = residual[np.isfinite(residual)]
    mean, median, std = np.mean(residual), np.median(residual), np.std(residual)
    fig, ax = plt.subplots()
    ax.hist(residual, bins=40)
    ax.text(0.95, 0.95, f'mean = {mean:.3f}', horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
    ax.text(0.95, 0.91, f'median = {median:.3f}', horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
    ax.text(0.95, 0.87, f'std = {std:.3f}', horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
    ax.set_title('Distribution of Diffusion Model Scores')
    ax.set_ylabel('diffusion model score')
    ax.set_xlabel(f'number of samples (out of {samples.shape[0]})')
    if return_metrics: return fig, ax, mean, median, std
    return fig, ax

def constraint_metric(samples, constraint, return_metrics=False):
    samples = samples[-1] # only consider last sample, should be shape [1000, 2] or similar
    values = np.abs(constraint.constraint(torch.from_numpy(samples).float()).numpy())
    if np.isnan(values).any() or np.isinf(values).any():
        print('warning: constraint metric has nan or inf. cleaning this out')
        values = values[np.isfinite(values)]
    mean, median, std = np.mean(values), np.median(values), np.std(values)
    fig, ax = plt.subplots()
    ax.hist(values, bins=40)
    ax.text(0.95, 0.95, f'mean = {mean:.3f}', horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
    ax.text(0.95, 0.91, f'median = {median:.3f}', horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
    ax.text(0.95, 0.87, f'std = {std:.3f}', horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
    ax.set_title('Distribution of Constraint Values')
    ax.set_ylabel('constraint value $g(x)$')
    ax.set_xlabel(f'number of samples (out of {samples.shape[0]})')
    if return_metrics: return fig, ax, mean, median, std
    return fig, ax