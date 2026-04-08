from typing import Sequence, Union

import numpy as np


ArrayLike = Union[Sequence[float], np.ndarray]


def mean_diff(x: ArrayLike) -> float:
    return np.mean(np.diff(np.sort(x)))

def std_diff(x: ArrayLike) -> float:
    return np.std(np.diff(np.sort(x)))

def min_diff(x: ArrayLike) -> float:
    return np.min(np.diff(np.sort(x)))

def max_diff(x: ArrayLike) -> float:
    return np.max(np.diff(np.sort(x)))

def nth_smallest(x: np.ndarray, n: int) -> float:
    return np.partition(x, n - 1)[n - 1]

def second_min(x: np.ndarray) -> float:
    return nth_smallest(x, 2)
