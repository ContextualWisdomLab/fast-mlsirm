import numpy as np

def slow_dist(x, z):
    diff = x[None, :, :] - z[:, None, :]
    return np.sqrt(1e-12 + np.sum(diff * diff, axis=2))

def fast_dist_einsum(x, z, eps=1e-12):
    sq_x = np.einsum('ij,ij->i', x, x)
    sq_z = np.einsum('ij,ij->i', z, z)
    return np.sqrt(np.maximum(eps, sq_z[:, None] - 2 * np.dot(z, x.T) + sq_x[None, :]))

x = np.random.randn(250, 2)
z = np.random.randn(12, 2)

d1 = slow_dist(x, z)
d2 = fast_dist_einsum(x, z)
print("max err:", np.max(np.abs(d1 - d2)))
