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

    argv_sim = ["simulate", "--persons", "10", "--dims", "1", "--items-per-dim", "2", "--out", str(out_dir)]
    assert main(argv_sim) == 0

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

def test_main_block():
    import runpy
    import fast_mlsirm.cli
    import sys

    # We test running as main
    original_argv = sys.argv
    sys.argv = ["fast-mlsirm", "simulate", "--persons", "1", "--dims", "1", "--items-per-dim", "1", "--out", "test_out"]
    try:
        with pytest.raises(SystemExit) as excinfo:
            runpy.run_module("fast_mlsirm.cli", run_name="__main__")
        assert excinfo.value.code == 0
    finally:
        sys.argv = original_argv
        import shutil
        import os
        if os.path.exists("test_out"):
            shutil.rmtree("test_out")
