"""Rust <-> numpy numerical parity gate.

Rust is the *primary* numeric path for fast-mlsirm; the pure-numpy
implementation in :mod:`fast_mlsirm.objective` is the reference kept for
parity testing and as a fallback when the compiled ``fast_mlsirm._core``
extension is unavailable.

These tests assert that the Rust core reproduces the reference LSIRM/MLS2PLM
neg-loglik + gradients to a tight ``1e-6`` absolute tolerance over
representative fixtures (all five model variants, several problem sizes,
with and without missing-data masks). This is the acceptance gate for the
"move numeric computation to Rust" change: the observable outputs must not
change when the Rust backend takes over.
"""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm import FitConfig, MLSIRMParams
from fast_mlsirm.objective import neg_loglik_and_grad

# Tight parity tolerance required by the port: Rust f64 must match the numpy
# reference to within 1e-6 (in practice the difference is ~1e-13). This is an
# f64 contract, so the tight tests pin rust_device="cpu": on GPU hosts the
# default "auto" routes through the f32 wgpu kernels (~7e-6 error). The f32
# GPU/auto path has a separate 1e-4 model matrix below.
PARITY_ATOL = 1e-6
GPU_ATOL = 1e-4

MODELS = ["MLS2PLM", "MLSRM", "MIRT", "ULS2PLM", "ULSRM"]

# (n_persons, n_items, n_dims, latent_dim) fixture shapes. n_dims is clamped to
# 1 for the unidimensional (ULS*) models inside the builder.
SHAPES = [
    (8, 4, 2, 2),
    (40, 12, 3, 2),
    (25, 6, 1, 1),
    (60, 15, 5, 3),
]


def _make_params(
    rng: np.random.Generator, n_persons: int, n_items: int, n_dims: int, latent_dim: int
) -> MLSIRMParams:
    return MLSIRMParams(
        theta=rng.normal(size=(n_persons, n_dims)),
        alpha=rng.normal(size=n_items) * 0.3,
        b=rng.normal(size=n_items),
        xi=rng.normal(size=(n_persons, latent_dim)),
        zeta=rng.normal(size=(n_items, latent_dim)),
        tau=0.35,
    )


def _fixture(model: str, shape: tuple[int, int, int, int], seed: int):
    n_persons, n_items, n_dims, latent_dim = shape
    if model in {"ULS2PLM", "ULSRM"}:
        n_dims = 1
    rng = np.random.default_rng(seed)
    params = _make_params(rng, n_persons, n_items, n_dims, latent_dim)
    if n_dims > 1:
        factors = (np.arange(n_items) % n_dims).astype(np.int64)
    else:
        factors = np.zeros(n_items, dtype=np.int64)
    y = (rng.random((n_persons, n_items)) < 0.5).astype(np.float64)
    return params, factors, y, n_persons, n_items


def _max_grad_diff(grad_a: MLSIRMParams, grad_b: MLSIRMParams) -> float:
    return max(
        float(np.max(np.abs(grad_a.theta - grad_b.theta))),
        float(np.max(np.abs(grad_a.alpha - grad_b.alpha))),
        float(np.max(np.abs(grad_a.b - grad_b.b))),
        float(np.max(np.abs(grad_a.xi - grad_b.xi))),
        float(np.max(np.abs(grad_a.zeta - grad_b.zeta))),
        float(abs(grad_a.tau - grad_b.tau)),
    )


@pytest.fixture(autouse=True)
def _require_rust_core():
    pytest.importorskip("fast_mlsirm._core")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("shape", SHAPES)
def test_rust_matches_numpy_dense(model: str, shape: tuple[int, int, int, int]) -> None:
    params, factors, y, _, _ = _fixture(model, shape, seed=hash((model, shape)) % 2**32)
    config = FitConfig(model=model, max_iter=1, rust_device="cpu")

    n_obj, n_grad, n_ll = neg_loglik_and_grad(y, factors, params, config, backend="numpy")
    r_obj, r_grad, r_ll = neg_loglik_and_grad(y, factors, params, config, backend="rust")

    assert abs(r_obj - n_obj) < PARITY_ATOL
    assert abs(r_ll - n_ll) < PARITY_ATOL
    assert _max_grad_diff(r_grad, n_grad) < PARITY_ATOL


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("shape", SHAPES)
def test_rust_matches_numpy_with_mask(model: str, shape: tuple[int, int, int, int]) -> None:
    params, factors, y, n_persons, n_items = _fixture(
        model, shape, seed=(hash((model, shape, "mask")) % 2**32)
    )
    rng = np.random.default_rng(7)
    mask = rng.random((n_persons, n_items)) < 0.85
    # Guarantee no all-missing row/column (rejected by prepare_response).
    mask[:, 0] = True
    mask[0, :] = True
    config = FitConfig(model=model, max_iter=1, rust_device="cpu")

    n_obj, n_grad, n_ll = neg_loglik_and_grad(y, factors, params, config, mask=mask, backend="numpy")
    r_obj, r_grad, r_ll = neg_loglik_and_grad(y, factors, params, config, mask=mask, backend="rust")

    assert abs(r_obj - n_obj) < PARITY_ATOL
    assert abs(r_ll - n_ll) < PARITY_ATOL
    assert _max_grad_diff(r_grad, n_grad) < PARITY_ATOL


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("device", ["gpu", "auto"])
@pytest.mark.parametrize("masked", [False, True])
def test_rust_gpu_or_auto_matches_numpy_model_matrix(
    model: str, device: str, masked: bool
) -> None:
    params, factors, y, n_persons, n_items = _fixture(
        model, (18, 9, 3, 2), seed=hash((model, device, masked, "gpu")) % 2**32
    )
    mask = None
    if masked:
        rng = np.random.default_rng(17)
        mask = rng.random((n_persons, n_items)) < 0.8
        mask[:, 0] = True
        mask[0, :] = True
    config = FitConfig(model=model, max_iter=1, rust_device=device)

    n_obj, n_grad, n_ll = neg_loglik_and_grad(
        y, factors, params, config, mask=mask, backend="numpy"
    )
    r_obj, r_grad, r_ll = neg_loglik_and_grad(
        y, factors, params, config, mask=mask, backend="rust"
    )

    assert abs(r_obj - n_obj) < GPU_ATOL
    assert abs(r_ll - n_ll) < GPU_ATOL
    assert _max_grad_diff(r_grad, n_grad) < GPU_ATOL


def test_rust_is_default_resolved_backend() -> None:
    """With the extension built, the default ("auto") backend resolves to Rust."""
    from fast_mlsirm.backend import resolve_backend

    assert resolve_backend(FitConfig().backend) == "rust"
