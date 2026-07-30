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
    Compute pairwise distances between xi and zeta.
    xi shape: (N, D)
    zeta shape: (I, D)
    Returns: (I, N) distance matrix.
    """
    max_val = max(np.max(np.abs(xi)) if xi.size > 0 else 0, np.max(np.abs(zeta)) if zeta.size > 0 else 0)
    if max_val > 1e100:
        # Fallback for numerical stability
        diff = xi[None, :, :] - zeta[:, None, :]
        return np.sqrt(eps_distance + np.sum(diff * diff, axis=2))

    sq_xi = np.einsum('ij,ij->i', xi, xi)
    sq_zeta = np.einsum('ij,ij->i', zeta, zeta)
    return np.sqrt(np.maximum(eps_distance, sq_zeta[:, None] - 2 * np.dot(zeta, xi.T) + sq_xi[None, :]))


def pairwise_distance_1d(xi: np.ndarray, zeta_c: np.ndarray, eps_distance: float = 1e-12) -> np.ndarray:
    """
    Compute pairwise distances between xi and a single zeta vector.
    xi shape: (N, D)
    zeta_c shape: (D,)
    Returns: (N,) distance vector.
    """
    max_val = max(np.max(np.abs(xi)) if xi.size > 0 else 0, np.max(np.abs(zeta_c)) if zeta_c.size > 0 else 0)
    if max_val > 1e100:
        # Fallback for numerical stability
        diff = xi - zeta_c[None, :]
        return np.sqrt(eps_distance + np.sum(diff * diff, axis=1))

    sq_xi = np.einsum('ij,ij->i', xi, xi)
    sq_zeta_c = np.einsum('i,i->', zeta_c, zeta_c)
    return np.sqrt(np.maximum(eps_distance, sq_zeta_c - 2 * np.dot(xi, zeta_c) + sq_xi))
