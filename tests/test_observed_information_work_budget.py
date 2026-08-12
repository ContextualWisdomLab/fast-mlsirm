"""Fail-first resource budgets for public observed-information evaluation."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm._core as core
import fast_mlsirm.inference as inference
from fast_mlsirm.config import FitConfig
from fast_mlsirm.types import MLSIRMParams


def _small_mirt_problem() -> tuple[np.ndarray, np.ndarray, MLSIRMParams]:
    """Return a tiny MIRT problem with a non-empty packed parameter vector."""
    responses = np.array([[1.0], [0.0]], dtype=np.float64)
    factor_id = np.array([0], dtype=np.int64)
    params = MLSIRMParams(
        theta=np.array([[0.25], [-0.25]], dtype=np.float64),
        alpha=np.array([0.1], dtype=np.float64),
        b=np.array([0.0], dtype=np.float64),
        xi=np.zeros((2, 1), dtype=np.float64),
        zeta=np.zeros((1, 1), dtype=np.float64),
        tau=-2.0,
    )
    return responses, factor_id, params


def _objective_must_not_run(*_args, **_kwargs):
    """Fail if resource preflight lets the expensive objective start."""
    raise AssertionError("objective evaluated before observed-information preflight")


def _constant_objective(*_args, **_kwargs):
    """Return a cheap finite objective so allocation behavior can be isolated."""
    return 0.0, None, None


def test_observed_information_rejects_objective_call_budget_before_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact ``1 + 2*n**2`` call budget must fail closed before objective work."""
    responses, factor_id, params = _small_mirt_problem()
    monkeypatch.setattr(
        inference,
        "_MAX_OBSERVED_INFORMATION_OBJECTIVE_CALLS",
        1,
        raising=False,
    )
    monkeypatch.setattr(
        inference,
        "_MAX_OBSERVED_INFORMATION_WORKSPACE_BYTES",
        2**63 - 1,
        raising=False,
    )
    monkeypatch.setattr(inference, "neg_loglik_and_grad", _objective_must_not_run)

    with pytest.raises(ValueError, match="objective-call budget"):
        inference.observed_information(
            responses,
            factor_id,
            params,
            config=FitConfig(model="MIRT", backend="rust"),
            backend="rust",
            device="cpu",
        )


def test_observed_information_rejects_workspace_budget_before_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dense finite-difference workspace must be budgeted before objective work."""
    responses, factor_id, params = _small_mirt_problem()
    monkeypatch.setattr(
        inference,
        "_MAX_OBSERVED_INFORMATION_OBJECTIVE_CALLS",
        2**63 - 1,
        raising=False,
    )
    monkeypatch.setattr(
        inference,
        "_MAX_OBSERVED_INFORMATION_WORKSPACE_BYTES",
        1,
        raising=False,
    )
    monkeypatch.setattr(inference, "neg_loglik_and_grad", _objective_must_not_run)

    with pytest.raises(ValueError, match="workspace budget"):
        inference.observed_information(
            responses,
            factor_id,
            params,
            config=FitConfig(model="MIRT", backend="rust"),
            backend="rust",
            device="cpu",
        )


def test_observed_information_avoids_dense_identity_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Coordinate perturbations must not require a second dense ``n x n`` identity."""
    responses, factor_id, params = _small_mirt_problem()
    monkeypatch.setattr(
        inference,
        "_MAX_OBSERVED_INFORMATION_OBJECTIVE_CALLS",
        2**63 - 1,
        raising=False,
    )
    monkeypatch.setattr(
        inference,
        "_MAX_OBSERVED_INFORMATION_WORKSPACE_BYTES",
        2**63 - 1,
        raising=False,
    )
    monkeypatch.setattr(inference, "neg_loglik_and_grad", _constant_objective)

    def forbidden_eye(*_args, **_kwargs):
        raise AssertionError("observed_information materialized a dense identity matrix")

    monkeypatch.setattr(inference.np, "eye", forbidden_eye)
    monkeypatch.setattr(
        core,
        "observed_information",
        lambda n, *_args, **_kwargs: [0.0] * (int(n) * int(n)),
    )

    result = inference.observed_information(
        responses,
        factor_id,
        params,
        config=FitConfig(model="MIRT", backend="rust"),
        backend="rust",
        device="cpu",
    )

    assert result.shape[0] == result.shape[1]
    assert np.all(result == 0.0)
