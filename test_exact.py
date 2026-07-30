import numpy as np
import time

def slow_dist(x, z):
    diff = x[None, :, :] - z[:, None, :]
    return np.sqrt(1e-12 + np.sum(diff * diff, axis=2))

def fast_dist_exact(x, z):
    dist = np.zeros((z.shape[0], x.shape[0]), dtype=np.float64)
    for k in range(x.shape[1]):
        diff = x[:, k][None, :] - z[:, k][:, None]
        dist += diff * diff
    return np.sqrt(1e-12 + dist)

def fast_dist_einsum(x, z):
    sq_x = np.einsum('ij,ij->i', x, x)
    sq_z = np.einsum('ij,ij->i', z, z)
    return np.sqrt(np.maximum(1e-12, sq_z[:, None] - 2 * np.dot(z, x.T) + sq_x[None, :]))

x = np.random.randn(5000, 2)
z = np.random.randn(200, 2)

d1 = slow_dist(x, z)
d2 = fast_dist_exact(x, z)
d3 = fast_dist_einsum(x, z)

print("Exact match (loop):", np.max(np.abs(d1 - d2)))
print("Einsum match:", np.max(np.abs(d1 - d3)))

t0 = time.time()
for _ in range(100): slow_dist(x, z)
t1 = time.time()
print("Slow:", t1-t0)

t0 = time.time()
for _ in range(100): fast_dist_exact(x, z)
t1 = time.time()
print("Exact:", t1-t0)

t0 = time.time()
for _ in range(100): fast_dist_einsum(x, z)
t1 = time.time()
print("Einsum:", t1-t0)
