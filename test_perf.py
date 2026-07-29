import numpy as np
import time

I = 100
Nx = 1000
K = 2
eps_distance = 1e-12

x_grid = np.random.randn(Nx, K)
zeta = np.random.randn(I, K)

start = time.perf_counter()
for _ in range(100):
    diff = x_grid[None, :, :] - zeta[:, None, :]
    dist1 = np.sqrt(eps_distance + np.sum(diff * diff, axis=2))
end = time.perf_counter()
print(f"Original: {end - start:.5f}s")

start = time.perf_counter()
for _ in range(100):
    x_sq = np.einsum('ij,ij->i', x_grid, x_grid)[None, :]
    zeta_sq = np.einsum('ij,ij->i', zeta, zeta)[:, None]
    dot = np.dot(zeta, x_grid.T)
    dist2 = np.sqrt(np.maximum(eps_distance + x_sq - 2 * dot + zeta_sq, eps_distance))
end = time.perf_counter()
print(f"Optimized: {end - start:.5f}s")
print(f"Max diff: {np.abs(dist1 - dist2).max()}")
