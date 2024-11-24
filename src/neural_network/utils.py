# neural_network/utils.py

import numpy as np


def initialize_weights(input_size: int, output_size: int) -> np.ndarray:
    ##xavier
    limit = np.sqrt(6 / (input_size + output_size))
    return np.random.uniform(-limit, limit, (input_size, output_size))


def orthogonal_init(input_size: int, output_size: int) -> np.ndarray:
    a = np.random.randn(input_size, output_size)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    return u if u.shape == (input_size, output_size) else v


def normalize_data(X: np.ndarray) -> np.ndarray:
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-12  ##zero
    return (X - mean) / std


def min_max_scale(X: np.ndarray, feature_range: tuple = (0, 1)) -> np.ndarray:

    min_val, max_val = feature_range
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    return (X - X_min) / (X_max - X_min + 1e-12) * (max_val - min_val) + min_val
