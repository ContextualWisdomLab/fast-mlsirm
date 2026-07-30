from __future__ import annotations

import numpy as np

from .types import MLSIRMParams


def sigmoid(x: np.ndarray) -> np.ndarray:
    x_safe = np.clip(x, -709.0, 709.0)
    return 1.0 / (1.0 + np.exp(-x_safe))


def softplus(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def logit(p: np.ndarray | float, eps: float = 1e-6) -> np.ndarray:
    clipped = np.clip(p, eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mean = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd < 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return (x - mean) / sd


def normalize_latent_positions(params: MLSIRMParams) -> MLSIRMParams:
    out = params.copy()
    if out.xi.size == 0 or out.zeta.size == 0:
        return out

    combined = np.vstack([out.xi, out.zeta])
    center = combined.mean(axis=0)
    out.xi -= center
    out.zeta -= center

    scale = float(np.std(np.vstack([out.xi, out.zeta])))
    if np.isfinite(scale) and scale > 1e-12:
        out.xi /= scale
        out.zeta /= scale
        out.tau += float(np.log(scale))
    return out


def pairwise_distance(xi: np.ndarray, zeta: np.ndarray, eps_distance: float = 1e-12) -> np.ndarray:
    """
    Compute pairwise distances between xi and zeta without allocating 3D intermediate arrays.
    xi shape: (N, D)
    zeta shape: (I, D)
    Returns: (I, N) distance matrix.
    """
    if xi.size == 0 or zeta.size == 0:
        return np.zeros((zeta.shape[0], xi.shape[0]), dtype=np.float64)

    D = xi.shape[1]
    dist = np.zeros((zeta.shape[0], xi.shape[0]), dtype=np.float64)
    for k in range(D):
        # (1, N) - (I, 1) -> (I, N)
        diff = xi[:, k][None, :] - zeta[:, k][:, None]
        dist += diff * diff
    return np.sqrt(eps_distance + dist)


def pairwise_distance_1d(xi: np.ndarray, zeta_c: np.ndarray, eps_distance: float = 1e-12) -> np.ndarray:
    """
    Compute pairwise distances between xi and a single zeta vector.
    xi shape: (N, D)
    zeta_c shape: (D,)
    Returns: (N,) distance vector.
    """
    if xi.size == 0 or zeta_c.size == 0:
        return np.zeros(xi.shape[0], dtype=np.float64)

    D = xi.shape[1]
    dist = np.zeros(xi.shape[0], dtype=np.float64)
    for k in range(D):
        diff = xi[:, k] - zeta_c[k]
        dist += diff * diff
    return np.sqrt(eps_distance + dist)
