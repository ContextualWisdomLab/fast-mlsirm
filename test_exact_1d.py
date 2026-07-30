import numpy as np
import time

def slow_dist(x, z):
    diff = x - z[None, :]
    return np.sqrt(1e-12 + np.sum(diff * diff, axis=1))

def fast_dist_exact(x, z):
    dist = np.zeros(x.shape[0], dtype=np.float64)
    for k in range(x.shape[1]):
        diff = x[:, k] - z[k]
        dist += diff * diff
    return np.sqrt(1e-12 + dist)

x = np.random.randn(5000, 2)
z = np.random.randn(2)

d1 = slow_dist(x, z)
d2 = fast_dist_exact(x, z)

print("Exact match (loop):", np.max(np.abs(d1 - d2)))
