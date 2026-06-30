import sys
import pytest
from io import StringIO
from fast_mlsirm.cli import main

def test_cli_simulate_invalid_args(monkeypatch):
    argv = ["simulate", "--persons", "notanint", "--out", "test_out"]
    with pytest.raises(SystemExit) as e:
        main(argv)
    assert e.value.code == 2

def test_cli_fit_missing_file(monkeypatch):
    argv = ["fit", "--responses", "non_existent.npy", "--factors", "whatever.csv", "--out", "out"]

    # We patch stderr to capture the output directly from sys.stderr
    captured_stderr = StringIO()
    monkeypatch.setattr(sys, "stderr", captured_stderr)

    code = main(argv)

    assert code == 1
    assert "❌ Error:" in captured_stderr.getvalue()
    assert "Traceback" not in captured_stderr.getvalue()

def test_cli_simulate_success(monkeypatch, tmp_path):
    out_dir = tmp_path / "sim_out"
    argv = ["simulate", "--persons", "10", "--dims", "1", "--items-per-dim", "2", "--out", str(out_dir)]
    code = main(argv)

    assert code == 0
    assert (out_dir / "responses.npy").exists()
    assert (out_dir / "manifest.json").exists()

def test_cli_fit_success(monkeypatch, tmp_path):
    out_dir = tmp_path / "sim_out"
    fit_out_dir = tmp_path / "fit_out"

    # Run simulate first to generate valid data
    argv_sim = ["simulate", "--persons", "10", "--dims", "1", "--items-per-dim", "2", "--out", str(out_dir)]
    assert main(argv_sim) == 0

    # Run fit with the generated data
    argv_fit = [
        "fit",
        "--responses", str(out_dir / "responses.npy"),
        "--factors", str(out_dir / "item_factor.csv"),
        "--out", str(fit_out_dir),
        "--max-iter", "1",
        "--n-restarts", "1"
    ]
    code = main(argv_fit)

    assert code == 0
    assert (fit_out_dir / "params.npz").exists()
    assert (fit_out_dir / "fit_summary.json").exists()
