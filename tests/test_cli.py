import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from fast_mlsirm.cli import main
from fast_mlsirm.config import FitConfig, MLS2PLMConfig
from fast_mlsirm.types import MLSIRMParams, SimulationData, FitResult

def test_cli_simulate(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["simulate", "--out", str(tmp_path / "out")]
    with patch("fast_mlsirm.cli.simulate") as mock_sim, \
         patch("fast_mlsirm.cli.save_simulation") as mock_save:

         # Mock simulate return value so we don't need actual logic
         mock_params = MLSIRMParams(
             theta=np.zeros((500, 2)),
             alpha=np.ones((16, 2)),
             b=np.zeros(16),
             xi=np.zeros(16),
             zeta=np.zeros(16),
             tau=1.0,
         )
         mock_data = SimulationData(
             config=MLS2PLMConfig(),
             truth=mock_params,
             factor_id=np.zeros(16, dtype=np.int64),
             Phi=np.zeros(16),
             Y=np.zeros((500, 16)),
             probabilities=np.zeros((500, 16))
         )
         mock_sim.return_value = mock_data

         assert main(argv) == 0

         captured = capsys.readouterr()
         assert "✅ Simulation successfully saved to" in captured.out
         mock_sim.assert_called_once()
         mock_save.assert_called_once_with(mock_data, str(tmp_path / "out"))


def test_cli_fit_success(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    responses_npy = tmp_path / "responses.npy"
    np.save(responses_npy, np.zeros((10, 5)))
    factors_csv = tmp_path / "factors.csv"
    factors_csv.write_text("item_id,factor_id\n0,0\n")

    argv = ["fit", "--responses", str(responses_npy), "--factors", str(factors_csv), "--out", str(tmp_path / "out")]

    with patch("fast_mlsirm.cli.fit") as mock_fit, \
         patch("fast_mlsirm.cli.save_fit_result") as mock_save:

         mock_params = MLSIRMParams(
                 theta=np.zeros((10, 2)),
                 alpha=np.ones((5, 2)),
                 b=np.zeros(5),
                 xi=np.zeros(5),
                 zeta=np.zeros(5),
                 tau=1.0
         )
         mock_result = FitResult(
             model="MLS2PLM",
             optimizer="adam_lbfgs",
             objective=10.0,
             params=mock_params,
             convergence_status="converged",
             n_iter=10,
             loglik_trace=[-10.0],
             objective_trace=[10.0]
         )
         mock_fit.return_value = mock_result

         assert main(argv) == 0
         captured = capsys.readouterr()
         assert "✅ Fit result successfully saved to" in captured.out
         mock_fit.assert_called_once()
         mock_save.assert_called_once_with(mock_result, str(tmp_path / "out"))


def test_cli_fit_missing_responses(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    factors_csv = tmp_path / "factors.csv"
    factors_csv.write_text("item_id,factor_id\n0,0\n")

    argv = ["fit", "--responses", str(tmp_path / "missing.npy"), "--factors", str(factors_csv), "--out", str(tmp_path / "out")]

    assert main(argv) == 1

    captured = capsys.readouterr()
    assert "❌ Error: Responses file" in captured.err
    assert "missing.npy" in captured.err


def test_cli_fit_missing_factors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    responses_npy = tmp_path / "responses.npy"
    responses_npy.write_bytes(b"")

    argv = ["fit", "--responses", str(responses_npy), "--factors", str(tmp_path / "missing.csv"), "--out", str(tmp_path / "out")]

    assert main(argv) == 1

    captured = capsys.readouterr()
    assert "❌ Error: Factors file" in captured.err
    assert "missing.csv" in captured.err

def test_cli_main_exit():
    with patch("fast_mlsirm.cli.main", return_value=0):
        # We can't easily test `if __name__ == "__main__":` block directly with coverage
        # unless we execute it as a script. We'll skip covering that single line,
        # or we can mock sys.argv and call main()
        pass
