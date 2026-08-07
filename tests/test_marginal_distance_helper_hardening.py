"""Adversarial type and branch contracts for marginal distance helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import fast_mlsirm.estimators.marginal as marginal


class DerivedArray(np.ndarray):
    """Untrusted NumPy subclass that must not cross the BLAS boundary."""


def _derived_array(shape: tuple[int, int]) -> DerivedArray:
    """Return one float64 ndarray subclass with the requested shape."""

    return np.zeros(shape, dtype=np.float64).view(DerivedArray)


def test_workspace_helper_rejects_empty_operation_name() -> None:
    """Resource errors require one stable nonempty operation name."""

    with pytest.raises(ValueError, match="name"):
        marginal._checked_marginal_workspace_bytes(
            "",
            2,
            3,
            itemsize=np.dtype(np.float64).itemsize,
            limit_bytes=1_000,
        )


def test_workspace_helper_accepts_zero_product_without_large_multiplication() -> None:
    """A zero-sized dimension short-circuits later astronomical dimensions safely."""

    assert (
        marginal._checked_marginal_workspace_bytes(
            "zero workspace",
            5,
            0,
            10**200,
            itemsize=np.dtype(np.float64).itemsize,
            limit_bytes=1_000,
        )
        == 0
    )


def test_pairwise_helper_rejects_nonarrays_and_ndarray_subclasses() -> None:
    """Only exact package-controlled ndarray storage reaches matrix multiplication."""

    valid = np.zeros((2, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="array"):
        marginal._pairwise_euclidean_distances(
            [[0.0, 0.0]],  # type: ignore[arg-type]
            valid,
            eps_distance=1e-8,
        )
    with pytest.raises(ValueError, match="exact NumPy arrays"):
        marginal._pairwise_euclidean_distances(
            _derived_array((2, 2)),
            valid,
            eps_distance=1e-8,
        )
    with pytest.raises(ValueError, match="exact NumPy arrays"):
        marginal._pairwise_euclidean_distances(
            valid,
            _derived_array((2, 2)),
            eps_distance=1e-8,
        )


def test_pairwise_helper_rejects_zero_latent_width() -> None:
    """A zero-width latent space cannot produce authoritative distances."""

    with pytest.raises(ValueError, match="latent dimension"):
        marginal._pairwise_euclidean_distances(
            np.zeros((2, 0), dtype=np.float64),
            np.zeros((3, 0), dtype=np.float64),
            eps_distance=1e-8,
        )


def test_pairwise_helper_rejects_nonfinite_computed_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-finite post-BLAS result fails before estimator use."""

    real_sqrt = np.sqrt

    def corrupting_sqrt(values: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
        """Corrupt the in-place square-root result without changing inputs."""

        result = real_sqrt(values, *args, **kwargs)
        output = kwargs.get("out")
        target = output if output is not None else result
        if isinstance(target, tuple):
            target = target[0]
        target[...] = np.inf
        return target

    monkeypatch.setattr(marginal.np, "sqrt", corrupting_sqrt)
    with pytest.raises(ValueError, match="finite"):
        marginal._pairwise_euclidean_distances(
            np.zeros((2, 2), dtype=np.float64),
            np.zeros((3, 2), dtype=np.float64),
            eps_distance=1e-8,
        )


def test_pairwise_helper_returns_owned_c_contiguous_float64() -> None:
    """The helper returns one deterministic owned matrix for downstream mutation."""

    result = marginal._pairwise_euclidean_distances(
        np.asfortranarray(
            np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
        ),
        np.asfortranarray(
            np.array([[1.0, 0.0], [3.0, 2.0]], dtype=np.float64)
        ),
        eps_distance=1e-8,
    )

    assert result.dtype == np.float64
    assert result.flags.c_contiguous
    assert result.flags.owndata
