import numpy as np
from fast_mlsirm import FitConfig, MLS2PLMConfig, recovery_report, simulate
from fast_mlsirm.fit import fit


def test_fit_pipeline_smoke():
    data = simulate(
        MLS2PLMConfig(n_persons=30, n_dims=2, items_per_dim=3, latent_dim=2, seed=11)
    )
    result = fit(
        data.Y,
        data.factor_id,
        config=FitConfig(
            model="MLS2PLM", optimizer="adam", max_iter=3, n_restarts=1, seed=11
        ),
    )

    assert result.params.theta.shape == (30, 2)
    assert result.params.alpha.shape == (6,)
    assert result.params.xi.shape == (30, 2)
    assert np.isfinite(result.objective)

    report = recovery_report(data.truth, result.params)
    assert "distance_rmse" in report.summary


def test_mirt_fit_sets_gamma_near_zero():
    data = simulate(
        MLS2PLMConfig(n_persons=20, n_dims=2, items_per_dim=2, gamma=0.0, seed=17)
    )
    result = fit(
        data.Y,
        data.factor_id,
        config=FitConfig(model="MIRT", optimizer="adam", max_iter=2, n_restarts=1),
    )
    assert result.params.gamma < 1e-10
