import numpy as np
import pandas as pd
import torch

from sklearn.datasets import make_moons
from torch.utils.data import TensorDataset


def moons_dataset(n=8000):
    X, _ = make_moons(n_samples=n, random_state=42, noise=0.03)
    X[:, 0] = (X[:, 0] + 0.3) * 2 - 1
    X[:, 1] = (X[:, 1] + 0.3) * 3 - 1
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def line_dataset(n=8000):
    rng = np.random.default_rng(42)
    x = rng.uniform(-0.5, 0.5, n)
    y = rng.uniform(-1, 1, n)
    X = np.stack((x, y), axis=1)
    X *= 4
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def circle_dataset(n=8000):
    rng = np.random.default_rng(42)
    x = np.round(rng.uniform(-0.5, 0.5, n)/2, 1)*2
    y = np.round(rng.uniform(-0.5, 0.5, n)/2, 1)*2
    norm = np.sqrt(x**2 + y**2) + 1e-10
    x /= norm
    y /= norm
    theta = 2 * np.pi * rng.uniform(0, 1, n)
    r = rng.uniform(0, 0.03, n)
    x += r * np.cos(theta)
    y += r * np.sin(theta)
    X = np.stack((x, y), axis=1)
    X *= 3
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def dino_dataset(n=8000, numpy=False):
    df = pd.read_csv("./static/DatasaurusDozen.tsv", sep="\t")
    df = df[df["dataset"] == "dino"]

    rng = np.random.default_rng(42)
    ix = rng.integers(0, len(df), n)
    x = df["x"].iloc[ix].tolist()
    x = np.array(x) + rng.normal(size=len(x)) * 0.15
    y = df["y"].iloc[ix].tolist()
    y = np.array(y) + rng.normal(size=len(x)) * 0.15
    x = (x/54 - 1) * 4
    y = (y/48 - 1) * 4
    X = np.stack((x, y), axis=1)
    if numpy: return X
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))

# new dataset: spiral
def spiral_dataset(n=8000, ordered=False):
    rng = np.random.default_rng(42)
    theta = 2 * np.pi + rng.uniform(0, 4 * np.pi, n)
    if ordered:
        theta = np.linspace(2 * np.pi, 6 * np.pi, n)
    r = theta/(3 * np.pi)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    X = np.stack((x, y), axis=1)
    return X if ordered else TensorDataset(torch.from_numpy(X.astype(np.float32))) 

# new dataset: two points
def two_points(n=8000):
    rng = np.random.default_rng(42)
    y = np.zeros(n)
    x = rng.choice(2, n, p=[0.1, 0.9])
    X = np.stack((x, y), axis=1)
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))

# new dataset: ring
def ring_dataset(n=8000, numpy=False):
    rng = np.random.default_rng(42)
    r = rng.uniform(1, 2, n)
    theta = rng.uniform(0, 2 * np.pi, n)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    X = np.stack((x, y), axis=1)
    if numpy: return X
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))

# new dataset: cubic constraint
def cubic_dataset(n=8000, numpy=False):
    rng = np.random.default_rng(42)
    x = rng.uniform(-2, 2, n)
    y = x ** 3 - x
    X = np.stack((x, y), axis=1)
    if numpy: return X
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))

def get_dataset(name, n=8000):
    if name == "moons":
        return moons_dataset(n)
    elif name == "dino":
        return dino_dataset(n)
    elif name == "line":
        return line_dataset(n)
    elif name == "circle":
        return circle_dataset(n)
    elif name == "spiral":
        return spiral_dataset(n)
    elif name == "two_points":
        return two_points(n)
    elif name == "ring":
        return ring_dataset(n)
    elif name == "cubic":
        return cubic_dataset(n)
    else:
        raise ValueError(f"Unknown dataset: {name}")
