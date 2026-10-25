"""DIF relative-model selection acceptance tests (synthetic evidence only)."""

from dataclasses import dataclass

import numpy as np
import pytest

from fast_mlsirm.dif_information_criteria import (
    compare_dif_information_criteria,
    dif_fit_evidence,
)


@dataclass
class NativeFitShape:
    """Synthetic fit-shaped evidence; never a hardware-execution oracle."""

    loglik_trace: np.ndarray
    n_parameters: int
    converged: bool = True
    slope_prior_mu: float | None = None
    slope_prior_sd: float | None = None


def test_aic_selects_augmented_through_fit_adapter():
    compact = dif_fit_evidence(
        NativeFitShape(np.array([-501.0, -500.0]), 10),
        n_persons=200,
        comparison_key="same-data-rows-estimator-grid",
        likelihood_kind="unpenalized_marginal_ml",
    )
    augmented = dif_fit_evidence(
        NativeFitShape(np.array([-499.0, -496.0]), 12),
        n_persons=200,
        comparison_key="same-data-rows-estimator-grid",
        likelihood_kind="unpenalized_marginal_ml",
    )
    result = compare_dif_information_criteria(compact, augmented, criterion="AIC")
    assert result.delta_ic == -4.0
    assert result.prefer_augmented is True
    assert result.compact_ic == 1020.0
    assert result.augmented_ic == 1016.0
    assert result.arithmetic_backend == "rust_cpu_f64"
    assert result.inference == "relative_model_selection"


def test_fit_adapter_rejects_map_prior_even_with_a_separate_likelihood_trace():
    fit = NativeFitShape(
        np.array([-501.0, -500.0]), 10, slope_prior_mu=0.0, slope_prior_sd=1.0
    )
    with pytest.raises(ValueError, match="MAP|prior"):
        dif_fit_evidence(
            fit,
            n_persons=200,
            comparison_key="same",
            likelihood_kind="unpenalized_marginal_ml",
        )


@pytest.mark.parametrize(
    "trace",
    [np.array([]), np.array([np.nan]), np.array([[1.0]])],
    ids=["empty", "nan", "matrix"],
)
def test_fit_adapter_rejects_invalid_final_likelihood(trace):
    with pytest.raises(ValueError, match="loglik"):
        dif_fit_evidence(
            NativeFitShape(trace, 10),
            n_persons=200,
            comparison_key="same",
            likelihood_kind="unpenalized_marginal_ml",
        )


def test_fit_adapter_preserves_native_receipt_without_claiming_mstep_gpu():
    fit = NativeFitShape(np.array([-500.0]), 10)
    fit.gpu_execution_used = True
    fit.gpu_backend = "fixture-Metal"
    fit.gpu_device_name = "fixture-device"
    fit.cpu_fallback_reason = None
    evidence = dif_fit_evidence(
        fit,
        n_persons=200,
        comparison_key="same",
        likelihood_kind="unpenalized_marginal_ml",
    )
    assert evidence.execution.gpu_execution_used is True
    assert evidence.execution.backend == "fixture-Metal"
    assert evidence.execution.device_name == "fixture-device"
    assert evidence.execution.likelihood_stage == "unknown"
    assert evidence.execution.m_step_stage == "unknown"
    # A synthetic receipt verifies transport only, not actual GPU execution.
    plain = dif_fit_evidence(
        NativeFitShape(np.array([-500.0]), 10),
        n_persons=200,
        comparison_key="same",
        likelihood_kind="unpenalized_marginal_ml",
    )
    assert plain.execution.gpu_execution_used is None
    assert plain.execution.backend is None


@pytest.mark.parametrize("value", [True, 2.5, -1], ids=["bool", "fraction", "negative"])
def test_integer_evidence_controls_are_not_silently_coerced(value):
    with pytest.raises(ValueError, match="n_parameters"):
        dif_fit_evidence(
            NativeFitShape(np.array([-500.0]), value),
            n_persons=200,
            comparison_key="same",
            likelihood_kind="unpenalized_marginal_ml",
        )


def test_nonconverged_evidence_and_missing_key_reject_before_comparison():
    with pytest.raises(ValueError, match="converged"):
        dif_fit_evidence(
            NativeFitShape(np.array([-500.0]), 10, converged=False),
            n_persons=200,
            comparison_key="same",
            likelihood_kind="unpenalized_marginal_ml",
        )
    with pytest.raises(ValueError, match="comparison_key"):
        dif_fit_evidence(
            NativeFitShape(np.array([-500.0]), 10),
            n_persons=200,
            comparison_key=" ",
            likelihood_kind="unpenalized_marginal_ml",
        )


def test_fallback_reason_survives_without_gpu_promotion():
    fit = NativeFitShape(np.array([-500.0]), 10)
    fit.gpu_execution_used = False
    fit.cpu_fallback_reason = "fixture-no-adapter"
    e = dif_fit_evidence(
        fit,
        n_persons=200,
        comparison_key="same",
        likelihood_kind="unpenalized_marginal_ml",
    )
    assert e.execution.gpu_execution_used is False
    assert e.execution.cpu_fallback_reason == "fixture-no-adapter"
    assert e.execution.likelihood_stage == "unknown"


@pytest.mark.parametrize("criterion", ["AIC", "BIC", "AICc", "SABIC", "CAIC"])
def test_all_criterion_names_execute_native_and_retain_execution_receipts(criterion):
    compact = dif_fit_evidence(
        NativeFitShape(np.array([-500.0]), 10),
        n_persons=200,
        comparison_key="same",
        likelihood_kind="unpenalized_marginal_ml",
    )
    augmented = dif_fit_evidence(
        NativeFitShape(np.array([-496.0]), 12),
        n_persons=200,
        comparison_key="same",
        likelihood_kind="unpenalized_marginal_ml",
    )
    r = compare_dif_information_criteria(compact, augmented, criterion=criterion)
    assert np.isfinite(r.compact_ic)
    assert np.isfinite(r.augmented_ic)
    assert r.delta_ic == r.augmented_ic - r.compact_ic
    assert r.prefer_augmented is (r.delta_ic < 0)
    assert r.compact_execution is compact.execution
    assert r.augmented_execution is augmented.execution


def test_dic_and_unknown_selector_raise_through_native_boundary():
    c = dif_fit_evidence(
        NativeFitShape(np.array([-500.0]), 10),
        n_persons=200,
        comparison_key="same",
        likelihood_kind="unpenalized_marginal_ml",
    )
    a = dif_fit_evidence(
        NativeFitShape(np.array([-496.0]), 12),
        n_persons=200,
        comparison_key="same",
        likelihood_kind="unpenalized_marginal_ml",
    )
    with pytest.raises(ValueError, match="posterior"):
        compare_dif_information_criteria(c, a, criterion="DIC")
    with pytest.raises(ValueError, match="criterion"):
        compare_dif_information_criteria(c, a, criterion="Holm")


def test_fit_adapter_requires_explicit_likelihood_kind_for_untagged_fit():
    fit = NativeFitShape(np.array([-500.0]), 10)
    with pytest.raises(ValueError, match="likelihood_kind"):
        dif_fit_evidence(fit, n_persons=200, comparison_key="same")
