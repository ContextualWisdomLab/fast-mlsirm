from __future__ import annotations

import numpy as np

from .types import MLSIRMParams


def sigmoid(x: np.ndarray) -> np.ndarray:
    # 0-d 배열과 dtype 유지를 위해 asarray로 캐스팅
    x_safe = np.asarray(np.clip(x, -709.0, 709.0))
    if x_safe.ndim == 0:
        # np.clip이 scalar를 반환할 수 있으므로, at least 1D로 변환하거나 reshape 사용
        x_safe = x_safe.reshape(-1)
        np.negative(x_safe, out=x_safe)
        np.exp(x_safe, out=x_safe)
        np.add(1.0, x_safe, out=x_safe)
        np.divide(1.0, x_safe, out=x_safe)
        return x_safe.reshape(())

    np.negative(x_safe, out=x_safe)
    np.exp(x_safe, out=x_safe)
    np.add(1.0, x_safe, out=x_safe)
    np.divide(1.0, x_safe, out=x_safe)
    return x_safe


def softplus(x: np.ndarray) -> np.ndarray:
    dtype = np.float32 if np.asarray(x).dtype == np.float32 else np.float64
    out = np.asarray(np.abs(x, dtype=dtype))

    if out.ndim == 0:
        out = out.reshape(-1)
        np.negative(out, out=out)
        np.exp(out, out=out)
        np.log1p(out, out=out)
        out += np.maximum(x, 0.0).reshape(-1)
        return out.reshape(())

    np.negative(out, out=out)
    np.exp(out, out=out)
    np.log1p(out, out=out)
    out += np.maximum(x, 0.0)
    return out


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
