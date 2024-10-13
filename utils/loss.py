from typing import Callable
import numpy as np


def atari_reward_transform(x: np.ndarray, var_eps: float = 0.001) -> np.ndarray:
    return np.sign(x) * (np.sqrt(np.abs(x) + 1) - 1) + var_eps * x


def inverse_atari_reward_transform(x: np.ndarray, var_eps: float = 0.001) -> np.ndarray:
    return np.sign(x) * (((np.sqrt(1 + 4 * var_eps * (np.abs(x) + 1 + var_eps)) - 1) / (2 * var_eps)) ** 2 - 1)


def scalar_to_support(x: np.ndarray, support_size: int, reward_transformer: Callable = atari_reward_transform, **kwargs) -> np.ndarray:
    if support_size == 0:  # Simple regression (support in this case can be the mean of a Gaussian)
        return x

    # Clip float to fit within the support_size. Values exceeding this will be assigned to the closest bin.
    transformed = np.clip(reward_transformer(x, **kwargs), a_min=-support_size, a_max=support_size - 1e-8)
    floored = np.floor(transformed).astype(int)  # Lower-bound support integer
    prob = transformed - floored                 # Proportion between adjacent integers

    bins = np.zeros((len(x), 2 * support_size + 1))

    bins[np.arange(len(x)), floored + support_size] = 1 - prob
    bins[np.arange(len(x)), floored + support_size + 1] = prob

    return bins


def support_to_scalar(x: np.ndarray, support_size: int, inv_reward_transformer: Callable = inverse_atari_reward_transform, **kwargs) -> np.ndarray:
    if support_size == 0:  # Simple regression (support in this case can be the mean of a Gaussian)
        return x

    bins = np.arange(-support_size, support_size + 1)
    y = np.dot(x, bins)

    value = inv_reward_transformer(y, **kwargs)

    return value
