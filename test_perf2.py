import numpy as np

I = 100
Nx = 1000
K = 2
eps_distance = 1e-12

x_grid = np.random.randn(Nx, K)
zeta = np.random.randn(I, K)

# Original
diff = x_grid[None, :, :] - zeta[:, None, :]
dist1 = np.sqrt(eps_distance + np.sum(diff * diff, axis=2))

# Optimized
x_sq = np.einsum('ij,ij->i', x_grid, x_grid)[None, :]
zeta_sq = np.einsum('ij,ij->i', zeta, zeta)[:, None]
dot = np.dot(zeta, x_grid.T)
dist2 = np.sqrt(np.maximum(x_sq - 2 * dot + zeta_sq, 0.0) + eps_distance)

print(np.max(np.abs(dist1 - dist2)))
