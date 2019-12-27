import numpy as np

def mean_diff(x):
    return np.mean(np.diff(np.sort(x)))

def std_diff(x):
    return np.std(np.diff(np.sort(x)))
    
def min_diff(x):
    return np.min(np.diff(np.sort(x)))
    
def max_diff(x):
    return np.max(np.diff(np.sort(x)))

def nth_smallest(x, n):
    return np.partition(x, n - 1)[n - 1]

def second_min(x):
    return nth_smallest(x, 2)
