"""Two-tier FIPC group person-score contracts.

These tests use a fit-like object so the scoring contract is executable even
when the optional Rust extension is unavailable on a development machine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from fast_mlsirm.two_tier_fipc import (
    execute_two_tier_fipc_group_person_score_payload,
    fit_and_score_two_tier_fipc_group_persons,
    fit_two_tier_grm_fipc,
    score_two_tier_fipc_group_persons,
    two_tier_reference_expected_score_moments,
)


@dataclass
class _Fit:
    a_primary: np.ndarray
    a_specific: np.ndarray
    threshold: np.ndarray
    phi: np.ndarray
    n_primary: int
    n_specific: int
    orthogonal_primary_identification: bool = True


def _fit() -> _Fit:
    return _Fit(
        a_primary=np.array(
            [[1.2, 0.0], [1.0, 0.0], [0.9, 0.6], [0.8, 0.5], [0.0, 1.1], [0.0, 0.9]],
            dtype=np.float64,
        ),
        a_specific=np.array([0.7, 0.5, 0.6, -0.4, 0.8, 0.6]),
        threshold=np.tile(np.array([1.4, 0.0, -1.3]), (6, 1)),
        phi=np.eye(2),
        n_primary=2,
        n_specific=2,
    )


SPECIFIC_MAP = np.array([0, 0, 1, 1, 0, 1], dtype=np.int64)
ANCHOR = np.array([True, True, False, False, True, False])
RESPONSES = np.array(
    [[0, 1, 2, 3, 1, 2], [1, 2, 1, 2, 2, 3], [2, 1, 3, 1, 0, 2]],
    dtype=np.int64,
)


def test_two_tier_fipc_score_is_explicit_g4w_acceptance_path() -> None:
    result = score_two_tier_fipc_group_persons(
        RESPONSES,
        _fit(),
        SPECIFIC_MAP,
        ANCHOR,
        q_primary=9,
        q_specific=11,
        focal_primary_mean=np.array([0.8, -0.35]),
        focal_primary_sd=np.array([1.25, 0.75]),
        reference_primary_mean=np.zeros(2),
        reference_primary_sd=np.ones(2),
        focal_specific_sd=np.array([1.3, 0.65]),
    )
    assert result.theta_primary_eap.shape == (RESPONSES.shape[0], 2)
    assert result.theta_primary_sd.shape == result.theta_primary_eap.shape
    assert result.expected_raw.shape == (RESPONSES.shape[0],)
    assert np.all(np.isfinite(result.expected_raw))
    assert np.all((result.expected_raw >= 0.0) & (result.expected_raw <= 18.0))
    assert result.reference_moments.n_anchor_items == int(ANCHOR.sum())


def test_focal_prior_is_preserved_in_eap_and_negative_nonunit_fixture() -> None:
    fit = _fit()
    negative = score_two_tier_fipc_group_persons(
        RESPONSES,
        fit,
        SPECIFIC_MAP,
        ANCHOR,
        q_primary=7,
        q_specific=9,
        focal_primary_mean=np.array([-1.0, -0.5]),
        focal_primary_sd=np.array([1.4, 0.6]),
    )
    unit = score_two_tier_fipc_group_persons(
        RESPONSES,
        fit,
        SPECIFIC_MAP,
        ANCHOR,
        q_primary=7,
        q_specific=9,
    )
    assert float(negative.theta_primary_eap[:, 0].mean()) < float(unit.theta_primary_eap[:, 0].mean())


def test_reference_moments_use_anchor_only_rows() -> None:
    fit = _fit()
    baseline = two_tier_reference_expected_score_moments(
        fit, SPECIFIC_MAP, ANCHOR, q_primary=7, q_specific=9
    )
    poisoned = _Fit(
        a_primary=fit.a_primary.copy(),
        a_specific=fit.a_specific.copy(),
        threshold=fit.threshold.copy(),
        phi=fit.phi.copy(),
        n_primary=2,
        n_specific=2,
    )
    poisoned.a_primary[~ANCHOR] = 8.0
    poisoned.a_specific[~ANCHOR] = -7.0
    poisoned.threshold[~ANCHOR] = np.array([4.0, 0.0, -4.0])
    changed = two_tier_reference_expected_score_moments(
        poisoned, SPECIFIC_MAP, ANCHOR, q_primary=7, q_specific=9
    )
    assert changed.mean == pytest.approx(baseline.mean, abs=1e-12)
    assert changed.variance == pytest.approx(baseline.variance, abs=1e-12)


def test_reference_specific_prior_is_separate_from_focal_specific_prior() -> None:
    fit = _fit()
    unit = score_two_tier_fipc_group_persons(
        RESPONSES,
        fit,
        SPECIFIC_MAP,
        ANCHOR,
        q_primary=7,
        q_specific=9,
        reference_specific_sd=np.array([0.7, 1.4]),
        focal_specific_sd=np.array([1.6, 0.5]),
    )
    changed_focal = score_two_tier_fipc_group_persons(
        RESPONSES,
        fit,
        SPECIFIC_MAP,
        ANCHOR,
        q_primary=7,
        q_specific=9,
        reference_specific_sd=np.array([0.7, 1.4]),
        focal_specific_sd=np.array([0.4, 1.8]),
    )
    assert changed_focal.reference_moments.mean == pytest.approx(
        unit.reference_moments.mean, abs=1e-12
    )


def test_correlated_phi_fails_closed_without_conditional_nuisance_id() -> None:
    fit = _fit()
    fit.phi = np.array([[1.0, 0.2], [0.2, 1.0]])
    with pytest.raises(ValueError, match="Phi == I"):
        score_two_tier_fipc_group_persons(
            RESPONSES, fit, SPECIFIC_MAP, ANCHOR, q_primary=5, q_specific=5
        )


def test_payload_contract_is_serializable_and_explicit() -> None:
    fit = _fit()
    payload = {
        "responses": RESPONSES.tolist(),
        "fit": {
            "a_primary": fit.a_primary,
            "a_specific": fit.a_specific,
            "threshold": fit.threshold,
            "phi": fit.phi,
            "n_primary": fit.n_primary,
            "n_specific": fit.n_specific,
            "orthogonal_primary_identification": fit.orthogonal_primary_identification,
        },
        "specific_map": SPECIFIC_MAP.tolist(),
        "anchor": ANCHOR.tolist(),
        "q_primary": 5,
        "q_specific": 5,
        "focal_primary_mean": [-0.4, 0.2],
        "focal_primary_sd": [1.1, 0.8],
    }
    out = execute_two_tier_fipc_group_person_score_payload(payload)
    assert out["family"] == "two_tier_fipc_group_person_score"
    assert out["model_scope"] == "two_tier_grm_fipc_orthogonal_primary"
    assert len(out["expected_raw"]) == RESPONSES.shape[0]


def test_fit_without_required_id_metadata_fails_closed() -> None:
    fit = _fit()
    del fit.phi
    with pytest.raises(TypeError, match="phi"):
        score_two_tier_fipc_group_persons(
            RESPONSES, fit, SPECIFIC_MAP, ANCHOR, q_primary=5, q_specific=5
        )


def test_fit_without_orthogonal_id_confirmation_fails_closed() -> None:
    fit = _fit()
    fit.orthogonal_primary_identification = False
    with pytest.raises(ValueError, match="orthogonal_primary_identification"):
        score_two_tier_fipc_group_persons(
            RESPONSES, fit, SPECIFIC_MAP, ANCHOR, q_primary=5, q_specific=5
        )


def test_estimator_to_score_path_uses_rust_fit_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    fit = _fit()
    calls: list[tuple[object, ...]] = []

    class _Core:
        @staticmethod
        def fit_two_tier_grm_fipc(*args: object) -> dict[str, object]:
            calls.append(args)
            return {
                "a_primary": fit.a_primary,
                "a_specific": fit.a_specific,
                "threshold": fit.threshold,
                "phi": fit.phi,
                "primary_mean": np.array([0.65, -0.25]),
                "primary_sd": np.array([1.2, 0.8]),
                "specific_sd": np.array([1.1, 0.7]),
                "theta_p_eap": np.zeros((RESPONSES.shape[0], 2)),
                "theta_p_sd": np.ones((RESPONSES.shape[0], 2)),
                "category_counts": np.zeros((6, 4), dtype=np.int64),
                "loglik_trace": np.array([-10.0]),
                "n_iter": 3,
                "converged": True,
                "termination_reason": "tolerance_met",
                "final_loglik_change": 1e-8,
                "n_parameters": 20,
                "orthogonal_primary_identification": True,
            }

    monkeypatch.setattr("fast_mlsirm.fitstats._core_module", lambda: _Core())
    result = fit_and_score_two_tier_fipc_group_persons(
        RESPONSES,
        np.asarray(fit.a_primary != 0.0),
        SPECIFIC_MAP,
        4,
        2,
        2,
        ANCHOR,
        fit.a_primary,
        fit.a_specific,
        fit.threshold,
        q_primary=5,
        q_specific=5,
        max_iter=20,
        tol=1e-6,
    )
    assert len(calls) == 1
    assert result.fit.primary_mean.tolist() == pytest.approx([0.65, -0.25])
    assert result.theta_primary_eap.shape == (RESPONSES.shape[0], 2)
    assert np.all(np.isfinite(result.expected_raw))


def test_estimator_symbol_absence_is_explicitly_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("fast_mlsirm.fitstats._core_module", lambda: object())
    fit = _fit()
    with pytest.raises(RuntimeError, match="fit_two_tier_grm_fipc"):
        fit_two_tier_grm_fipc(
            RESPONSES,
            np.asarray(fit.a_primary != 0.0),
            SPECIFIC_MAP,
            4,
            2,
            2,
            ANCHOR,
            fit.a_primary,
            fit.a_specific,
            fit.threshold,
            q_primary=5,
            q_specific=5,
            max_iter=20,
            tol=1e-6,
        )
