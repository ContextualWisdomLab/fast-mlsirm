1. **Understand the Bottleneck**: In `fast_mlsirm/estimators/marginal.py` and `fast_mlsirm/fitstats.py`, there are several locations where Euclidean distance between matrices (like `x_grid` and `zeta`) is calculated using 3D broadcasting: `diff = x_grid[None, :, :] - zeta[:, None, :]` and `np.sqrt(eps_distance + np.sum(diff * diff, axis=2))`. This creates a massive 3D intermediate array `diff` of size `(I, Nx, K)`. Memory allocation for this large intermediate array slows down calculations substantially.

2. **The Optimization**: Instead of broadcasting and allocating the huge 3D array `diff`, we can utilize algebraic expansion and BLAS operations (dot product).
   For pairwise distances, `(a-b)^2 = a^2 - 2ab + b^2`.
   So, `np.sum((a - b)**2)` can be replaced with `a^2 + b^2 - 2ab`.
   Using `np.einsum` to quickly get the squared norms `x_sq` and `zeta_sq`, and `np.dot` to efficiently compute the cross term, avoids the 3D allocation and speeds up distance computation.

   To avoid precision issues where `a^2 + b^2 - 2ab` might be slightly negative, we use `np.maximum(..., 0.0)`. Thus, we calculate:
   ```python
   x_sq = np.einsum("ij,ij->i", x, x)[None, :]
   zeta_sq = np.einsum("ij,ij->i", zeta, zeta)[:, None]
   dist = np.sqrt(np.maximum(x_sq - 2 * np.dot(zeta, x.T) + zeta_sq, 0.0) + eps_distance)
   ```

3. **Locations to change**:
   - `python/fast_mlsirm/estimators/marginal.py`: lines 221-222, 765-766, 820-821
   - `python/fast_mlsirm/fitstats.py`: lines 270-271, 687-688, 766-767

   *Note*: Lines 692-693 and 723-724 in `marginal.py` compute distance with respect to `zeta_i` (1D or 2D). Need to carefully check the dimensions in those lines to correctly apply or leave as is if they are not massive (e.g., if `diff` is just `(Nx, K)`). Let's check `zeta_c` and `zeta_i` shapes first.
For `diff = x_grid - zeta_c[None, :]` and `diff = x_grid - zeta_i[None, :]`, these are 2D arrays (shape Nx, K) not 3D, because they are evaluated per item (for a single item). The allocation here is only (Nx, K) instead of (I, Nx, K).
However, for `x_grid[None, :, :] - zeta[:, None, :]`, the allocation is `(I, Nx, K)` which for N=1000, J=100, K=2 is large (200,000 items, and can grow quickly with I and Nx).
Also in `fitstats.py`: `diff = x_grid[None, :, :] - params.zeta[:, None, :]` creates a `(I, Nx, K)` array.

So we should optimize the 3D distance calculations in `estimators/marginal.py` and `fitstats.py`.

Let's do a quick code check:
In `fast_mlsirm/estimators/marginal.py` around line 221:
```python
<<<<<<< SEARCH
    if kind == "distance":
        diff = x_grid[None, :, :] - zeta[:, None, :]  # (I, Nx, K)
        dist = np.sqrt(eps_distance + np.sum(diff * diff, axis=2))  # (I, Nx)
        eta = eta - np.exp(tau) * dist[None, :, None, :]
=======
    if kind == "distance":
        # Bolt: Avoid allocating large 3D array (I, Nx, K) by using 2D dot product for pairwise Euclidean distance
        x_sq = np.einsum("ij,ij->i", x_grid, x_grid)[None, :]
        zeta_sq = np.einsum("ij,ij->i", zeta, zeta)[:, None]
        dist = np.sqrt(np.maximum(x_sq - 2 * np.dot(zeta, x_grid.T) + zeta_sq, 0.0) + eps_distance)
        eta = eta - np.exp(tau) * dist[None, :, None, :]
>>>>>>> REPLACE
```

In `fast_mlsirm/estimators/marginal.py` around line 765:
```python
<<<<<<< SEARCH
        if uses_space and anchor_tau is None and _interaction_kind(model) == "distance":
            gamma = float(np.exp(tau))
            diff = x_grid[None, :, :] - zeta[:, None, :]
            dist = np.sqrt(eps_distance + np.sum(diff * diff, axis=2))  # (I, Nx)
            a_all = np.exp(alpha) if free_alpha else np.ones(n_items)
=======
        if uses_space and anchor_tau is None and _interaction_kind(model) == "distance":
            gamma = float(np.exp(tau))
            # Bolt: Avoid allocating large 3D array (I, Nx, K) by using 2D dot product for pairwise Euclidean distance
            x_sq = np.einsum("ij,ij->i", x_grid, x_grid)[None, :]
            zeta_sq = np.einsum("ij,ij->i", zeta, zeta)[:, None]
            dist = np.sqrt(np.maximum(x_sq - 2 * np.dot(zeta, x_grid.T) + zeta_sq, 0.0) + eps_distance)
            a_all = np.exp(alpha) if free_alpha else np.ones(n_items)
>>>>>>> REPLACE
```

In `fast_mlsirm/estimators/marginal.py` around line 819:
```python
<<<<<<< SEARCH
            if kind_i == "distance":
                diffz = x_grid[None, :, :] - zeta[:, None, :]
                distz = np.sqrt(eps_distance + np.sum(diffz * diffz, axis=2))  # (I, Nx)
                interaction_term = -gamma * distz[None, :, None, :]
=======
            if kind_i == "distance":
                # Bolt: Avoid allocating large 3D array (I, Nx, K) by using 2D dot product for pairwise Euclidean distance
                x_sq = np.einsum("ij,ij->i", x_grid, x_grid)[None, :]
                zeta_sq = np.einsum("ij,ij->i", zeta, zeta)[:, None]
                distz = np.sqrt(np.maximum(x_sq - 2 * np.dot(zeta, x_grid.T) + zeta_sq, 0.0) + eps_distance)
                interaction_term = -gamma * distz[None, :, None, :]
>>>>>>> REPLACE
```

In `fast_mlsirm/fitstats.py` around line 270:
```python
<<<<<<< SEARCH
    if uses_space:
        diff = x_grid[None, :, :] - params.zeta[:, None, :]
        dist = np.sqrt(eps_distance + np.sum(diff * diff, axis=2))  # (I, Nx)
        eta = eta - math.exp(params.tau) * dist[:, None, :]
=======
    if uses_space:
        # Bolt: Avoid allocating large 3D array (I, Nx, K) by using 2D dot product for pairwise Euclidean distance
        x_sq = np.einsum("ij,ij->i", x_grid, x_grid)[None, :]
        zeta_sq = np.einsum("ij,ij->i", params.zeta, params.zeta)[:, None]
        dist = np.sqrt(np.maximum(x_sq - 2 * np.dot(params.zeta, x_grid.T) + zeta_sq, 0.0) + eps_distance)
        eta = eta - math.exp(params.tau) * dist[:, None, :]
>>>>>>> REPLACE
```

In `fast_mlsirm/fitstats.py` around line 687:
```python
<<<<<<< SEARCH
    if uses_space:
        diff = np.asarray(params.xi)[:, None, :] - np.asarray(params.zeta)[None, :, :]
        dist = np.sqrt(eps_distance + np.sum(diff * diff, axis=2))
        eta = eta - math.exp(params.tau) * dist
=======
    if uses_space:
        # Bolt: Avoid allocating large 3D array (N, J, K) by using 2D dot product for pairwise Euclidean distance
        xi = np.asarray(params.xi)
        zeta = np.asarray(params.zeta)
        xi_sq = np.einsum("ij,ij->i", xi, xi)[:, None]
        zeta_sq = np.einsum("ij,ij->i", zeta, zeta)[None, :]
        dist = np.sqrt(np.maximum(xi_sq - 2 * np.dot(xi, zeta.T) + zeta_sq, 0.0) + eps_distance)
        eta = eta - math.exp(params.tau) * dist
>>>>>>> REPLACE
```

In `fast_mlsirm/fitstats.py` around line 766:
```python
<<<<<<< SEARCH
    if uses_space:
        diff = np.asarray(params.xi)[:, None, :] - np.asarray(params.zeta)[None, :, :]
        dist = np.sqrt(eps_distance + np.sum(diff * diff, axis=2))
        eta = eta - math.exp(params.tau) * dist
=======
    if uses_space:
        # Bolt: Avoid allocating large 3D array (N, J, K) by using 2D dot product for pairwise Euclidean distance
        xi = np.asarray(params.xi)
        zeta = np.asarray(params.zeta)
        xi_sq = np.einsum("ij,ij->i", xi, xi)[:, None]
        zeta_sq = np.einsum("ij,ij->i", zeta, zeta)[None, :]
        dist = np.sqrt(np.maximum(xi_sq - 2 * np.dot(xi, zeta.T) + zeta_sq, 0.0) + eps_distance)
        eta = eta - math.exp(params.tau) * dist
>>>>>>> REPLACE
```
