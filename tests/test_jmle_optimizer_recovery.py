"""True-parameter recovery evidence for Rust-owned public JMLE optimizers.

The optimizer-ownership migration is not accepted scientifically from delegation
sentinels alone.  This study generates one deterministic, correctly specified
unidimensional MIRT data set, estimates it through each advertised public JMLE
optimizer, and checks convergence plus aligned bias/MAE/RMSE recovery.  The
thresholds are intentionally broad finite-sample acceptance bounds rather than
claims of asymptotic unbiasedness; JMLE retains the usual incidental-parameter
limitations.

References
----------
Harwell, M., Stone, C. A., Hsu, T.-C., & Kirisci, L. (1996). Monte Carlo
    studies in item response theory. *Applied Psychological Measurement,
    20*(2), 101-125. https://doi.org/10.1177/014662169602000201
Reckase, M. D. (2009). *Multidimensional item response theory*. Springer.
    https://doi.org/10.1007/978-0-387-89976-3
"""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm import FitConfig, MLS2PLMConfig, fit, recovery_report, simulate


def _mae(truth: np.ndarray, estimate: np.ndarray) -> float:
    """Return mean absolute error over identically identified parameters."""
    return float(np.mean(np.abs(np.asarray(estimate) - np.asarray(truth))))


def _standardize(values: np.ndarray) -> np.ndarray:
    """Standardize a parameter vector on its identified empirical scale."""
    array = np.asarray(values, dtype=np.float64)
    return (array - array.mean()) / array.std()


@pytest.fixture(scope="module")
def _jmle_recovery_data():
    """Return one deterministic 2PL-generating sample shared by optimizer modes."""
    return simulate(
        MLS2PLMConfig(
            n_persons=240,
            n_dims=1,
            items_per_dim=12,
            latent_dim=1,
            gamma=0.0,
            seed=62612,
        )
    )


@pytest.mark.parametrize("optimizer", ["adam", "lbfgs", "adam_lbfgs"])
def test_rust_jmle_optimizer_modes_recover_known_parameters(
    _jmle_recovery_data,
    optimizer: str,
) -> None:
    """Each public Rust JMLE optimizer must recover the same known-data scale."""
    data = _jmle_recovery_data
    result = fit(
        data.Y.astype(np.float64),
        data.factor_id,
        FitConfig(
            model="MIRT",
            estimator="jmle",
            optimizer=optimizer,
            latent_dim=1,
            max_iter=500,
            n_restarts=1,
            learning_rate=0.03,
            tolerance=1e-5,
            seed=62612,
            backend="rust",
            rust_device="cpu",
        ),
    )

    report = recovery_report(data.truth, result.params)
    metrics = report.metrics
    a_mae = _mae(data.truth.a, result.params.a)
    b_mae = _mae(data.truth.b, result.params.b)
    theta_mae = _mae(
        _standardize(data.truth.theta),
        _standardize(result.params.theta),
    )

    evidence = {
        "optimizer": optimizer,
        "status": result.convergence_status,
        "n_iter": result.n_iter,
        "a_bias": metrics["a_bias"],
        "a_mae": a_mae,
        "a_rmse": metrics["a_rmse"],
        "b_bias": metrics["b_bias"],
        "b_mae": b_mae,
        "b_rmse": metrics["b_rmse"],
        "theta_mae_standardized": theta_mae,
        "theta_rmse_standardized": metrics["theta_rmse_standardized"],
    }

    assert result.backend == "rust", evidence
    assert result.rust_device == "cpu", evidence
    assert result.convergence_status == "converged", evidence
    assert result.n_iter > 0, evidence
    assert result.objective_trace and np.all(np.isfinite(result.objective_trace)), evidence
    assert result.objective_trace[-1] <= result.objective_trace[0], evidence

    # Finite-sample recovery gates with deliberately wide headroom.  They make
    # bias, MAE and RMSE explicit without pretending penalized JMLE is an
    # asymptotically unbiased item estimator.
    assert abs(metrics["a_bias"]) < 1.0, evidence
    assert a_mae < 1.2, evidence
    assert metrics["a_rmse"] < 1.5, evidence
    assert abs(metrics["b_bias"]) < 1.0, evidence
    assert b_mae < 1.2, evidence
    assert metrics["b_rmse"] < 1.5, evidence
    assert theta_mae < 1.1, evidence
    assert metrics["theta_rmse_standardized"] < 1.35, evidence
