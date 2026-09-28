import numpy as np
import pytest

from fast_mlsirm.polytomous import fit_lsirm_polytomous


def test_public_lsirm_termination_receipt_grm_and_gpcm():
    response_matrix = np.array([[0, 1], [2, 1], [1, 0], [2, 2]], dtype=np.int64)
    for model_name in ("grm", "gpcm"):
        budget_fit = fit_lsirm_polytomous(
            response_matrix, 3, latent_dim=1, model=model_name, max_iter=1, tol=1e-12
        )
        assert not budget_fit.converged and budget_fit.termination_reason == "max_iter"
        assert budget_fit.n_iter == 1
        assert budget_fit.loglik_trace.size == budget_fit.n_iter + 1

        early_fit = fit_lsirm_polytomous(
            response_matrix, 3, latent_dim=1, model=model_name, max_iter=3, tol=1e6
        )
        assert early_fit.converged and early_fit.termination_reason == "tolerance"
        assert early_fit.n_iter < 3

        final_fit = fit_lsirm_polytomous(
            response_matrix, 3, latent_dim=1, model=model_name, max_iter=1, tol=1e6
        )
        assert final_fit.converged and final_fit.n_iter == 1
        for fitted_result in (budget_fit, early_fit, final_fit):
            assert fitted_result.loglik_trace.size == fitted_result.n_iter + 1
            assert np.all(np.isfinite(fitted_result.loglik_trace))
            assert np.isfinite(fitted_result.final_delta)
            assert np.isfinite(fitted_result.stopping_tolerance)
            assert fitted_result.final_delta == fitted_result.loglik_trace[-1] - fitted_result.loglik_trace[-2]
            assert fitted_result.loglik == fitted_result.loglik_trace[-1]


def test_public_lsirm_accepts_missing_cells_and_direct_binding_rejects_invalid_tolerance():
    missing_response_matrix = np.array(
        [[np.nan, np.nan], [0, np.nan], [1, 1]], dtype=np.float64
    )
    for model_name in ("grm", "gpcm"):
        fitted_result = fit_lsirm_polytomous(
            missing_response_matrix, 2, latent_dim=1, model=model_name, max_iter=1
        )
        assert fitted_result.loglik == fitted_result.loglik_trace[-1]
        assert np.all(np.isfinite(fitted_result.loglik_trace))

    from fast_mlsirm import _core

    response_vector = np.array([0], dtype=np.int64)
    for invalid_tolerance in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError):
            _core.fit_poly_lsirm(
                response_vector, 1, 1, 2, 1, None, "grm", 7, 7, 1, invalid_tolerance
            )
