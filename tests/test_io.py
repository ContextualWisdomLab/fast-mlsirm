import pytest
import numpy as np
from pathlib import Path
from fast_mlsirm.io import load_factor_csv, save_fit_result, load_params
from fast_mlsirm.types import FitResult, MLSIRMParams

def test_load_factor_csv_empty(tmp_path):
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("")
    with pytest.raises(ValueError, match="factor CSV is empty"):
        load_factor_csv(empty_csv)

def test_save_and_load_params(tmp_path):
    p = MLSIRMParams(
        theta=np.zeros((2, 2)),
        alpha=np.zeros(2),
        b=np.zeros(2),
        xi=np.zeros((2, 2)),
        zeta=np.zeros((2, 2)),
        tau=0.0
    )
    res = FitResult(
        params=p,
        model="MLS2PLM",
        optimizer="adam",
        objective=10.0,
        loglik_trace=[10.0, 9.0],
        objective_trace=[10.0, 9.0],
        convergence_status="converged",
        n_iter=2
    )
    out_dir = tmp_path / "fit"
    save_fit_result(res, out_dir)
    assert (out_dir / "params.npz").exists()
    assert (out_dir / "fit_summary.json").exists()

    loaded = load_params(out_dir / "params.npz")
    assert np.array_equal(loaded.theta, p.theta)
