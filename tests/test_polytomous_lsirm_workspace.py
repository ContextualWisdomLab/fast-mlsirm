"""Explicit core-workspace admission without allocating oversized grids."""

import inspect

import numpy as np
import pytest

from fast_mlsirm import _core
from fast_mlsirm.polytomous import fit_lsirm_polytomous


def test_workspace_budget_is_optional_beside_required_keywords():
    """ADR-0028 keeps model and quadrature required; the budget stays optional."""
    parameters = inspect.signature(fit_lsirm_polytomous).parameters
    for name in ("model", "q_theta", "q_xi", "max_iter", "tol"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters[name].default is inspect.Parameter.empty
    budget = parameters["workspace_budget_bytes"]
    assert budget.kind is inspect.Parameter.KEYWORD_ONLY
    assert budget.default is None


@pytest.mark.parametrize("model_family", ["grm", "gpcm"])
def test_workspace_budget_rejects_and_preserves_accepted_fit(model_family):
    response_matrix = np.array([[0, 1], [1, 0]], dtype=np.int64)
    fit_arguments = dict(
        n_cat=2,
        latent_dim=1,
        model=model_family,
        q_theta=7,
        q_xi=7,
        max_iter=1,
        tol=1e-5,
    )
    with pytest.raises(ValueError, match="workspace_budget_bytes"):
        fit_lsirm_polytomous(response_matrix, **fit_arguments, workspace_budget_bytes=1)
    legacy_fit = fit_lsirm_polytomous(response_matrix, **fit_arguments)
    admitted_fit = fit_lsirm_polytomous(
        response_matrix, **fit_arguments, workspace_budget_bytes=np.iinfo(np.uintp).max
    )
    assert admitted_fit.loglik == legacy_fit.loglik
    np.testing.assert_array_equal(admitted_fit.loglik_trace, legacy_fit.loglik_trace)
    np.testing.assert_array_equal(admitted_fit.cat_params, legacy_fit.cat_params)


@pytest.mark.parametrize("invalid_budget", [0, -1, True, np.bool_(True), 1.5, np.inf, "1024", 2**128])
def test_public_and_native_workspace_budget_reject_invalid_integer(invalid_budget):
    response_values = np.array([0], dtype=np.int64)
    with pytest.raises(ValueError, match="workspace_budget_bytes"):
        fit_lsirm_polytomous(
            response_values.reshape(1, 1),
            2,
            model="grm",
            q_theta=7,
            q_xi=7,
            max_iter=1,
            tol=1e-5,
            workspace_budget_bytes=invalid_budget,
        )
    with pytest.raises(ValueError, match="workspace_budget_bytes"):
        _core.fit_poly_lsirm(response_values, 1, 1, 2, 1,
                             workspace_budget_bytes=invalid_budget)
