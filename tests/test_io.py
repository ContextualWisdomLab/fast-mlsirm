import pytest
import numpy as np
from pathlib import Path
import json

from fast_mlsirm.io import load_factor_csv, load_params

def test_load_factor_csv_empty(tmp_path):
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("")

    with pytest.raises(ValueError, match="factor CSV is empty"):
        load_factor_csv(empty_csv)

def test_load_params(tmp_path):
    params_file = tmp_path / "params.npz"
    np.savez(
        params_file,
        theta=np.zeros((10, 2)),
        alpha=np.zeros(4),
        b=np.zeros(4),
        xi=np.zeros((10, 2)),
        zeta=np.zeros((4, 2)),
        tau=np.array(1.0)
    )

    params = load_params(params_file)
    assert params.theta.shape == (10, 2)
    assert params.tau == 1.0
