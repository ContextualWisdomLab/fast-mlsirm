from __future__ import annotations

import numpy as np

from .types import MLSIRMParams


def sigmoid(x: np.ndarray) -> np.ndarray:
    x_safe = np.clip(x, -709.0, 709.0)
    return 1.0 / (1.0 + np.exp(-x_safe))


def softplus(x: np.ndarray) -> np.ndarray:
    # Optimized softplus calculation: replace multiple intermediate array allocations
    # with a highly-optimized C ufunc np.logaddexp to improve execution speed.
    return np.logaddexp(x, 0.0)


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
