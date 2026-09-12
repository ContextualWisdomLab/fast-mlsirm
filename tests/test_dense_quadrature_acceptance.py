"""Acceptance checks for dense theta quadrature and guarded xi quadrature.

Small synthetic inputs cover rule admission and propagation only, not recovery
or numerical improvement. The fit path requires the compiled native extension.
"""

from itertools import product

import numpy as np
import pytest

import fast_mlsirm.polytomous as polytomous


@pytest.mark.parametrize("model_name", ["grm", "gpcm"])
def test_dense_rule_reaches_native_fit_and_scoring(model_name: str) -> None:
    """Fit accepts 61 nodes; missing-data scoring recovers the normal prior."""
    response_values = np.asarray(list(product(range(3), repeat=3)), dtype=float)
    # One EM iteration is sufficient for admission, not convergence evidence.
    fitted_model = polytomous.fit_polytomous(
        response_values, 3, model=model_name, q_theta=61, max_iter=1
    )
    assert np.isfinite(fitted_model.loglik)
    scored_values = polytomous.score_polytomous(
        response_values, fitted_model, q_theta=np.int64(61)
    )
    for score_name in ("theta_eap", "theta_sd"):
        assert scored_values[score_name].shape == (27,)
        assert np.all(np.isfinite(scored_values[score_name]))
    assert np.all(scored_values["theta_sd"] > 0)
    prior_scores = polytomous.score_polytomous(
        np.full((1, 3), np.nan), fitted_model, q_theta=61
    )
    np.testing.assert_allclose(prior_scores["theta_eap"], [0.0], atol=1e-12)
    np.testing.assert_allclose(prior_scores["theta_sd"], [1.0], atol=1e-12)


@pytest.mark.parametrize("quadrature_count", [61, np.int64(61)])
def test_dense_xi_rule_rejected_before_response_or_native_work(
    monkeypatch: pytest.MonkeyPatch, quadrature_count: int
) -> None:
    """The new trait rule cannot be admitted on the tensor-grid axis."""
    def forbidden_work(*call_args, **call_kwargs):
        """Fail if invalid xi reaches response conversion or core discovery."""
        raise AssertionError("invalid q_xi reached response or native work")

    monkeypatch.setattr(polytomous, "_poly_int_and_mask", forbidden_work)
    monkeypatch.setattr(polytomous, "_core_module", forbidden_work)
    with pytest.raises(ValueError, match="q_xi must be one of 7, 11, 15, 21, 31, 41"):
        polytomous.fit_lsirm_polytomous(
            object(), 3, q_theta=61, q_xi=quadrature_count
        )
