"""Fail-closed ordering tests for the NumPy MMLE fallback preflight."""

from __future__ import annotations

import pytest

from fast_mlsirm.estimators import mmle as mmle_module


class _UnexpectedArrayCoercion:
    """Array-like sentinel that fails if NumPy coercion is reached."""

    def __array__(self, *_args: object, **_kwargs: object) -> object:
        """Fail because invalid quadrature configuration must be rejected first."""
        raise AssertionError("response coercion ran before quadrature validation")


def test_invalid_quadrature_count_fails_before_response_array_coercion() -> None:
    """Invalid node counts must fail before potentially large response coercion."""
    sentinel = _UnexpectedArrayCoercion()

    with pytest.raises(ValueError, match=r"n_nodes must be"):
        mmle_module.fit_mmle_2pl(
            sentinel,  # type: ignore[arg-type]
            sentinel,  # type: ignore[arg-type]
            n_nodes=0,
            max_iter=1,
        )
