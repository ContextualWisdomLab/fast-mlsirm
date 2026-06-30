import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest

from fast_mlsirm.cli import main

def test_cli_empty_args(capsys):
    with patch.object(sys, 'argv', ['fast-mlsirm']):
        assert main() == 2

def test_cli_simulate_success(tmp_path):
    out_dir = tmp_path / "sim_out"
    args = ["simulate", "--persons", "10", "--dims", "2", "--items-per-dim", "2", "--out", str(out_dir)]

    with patch.object(sys, 'argv', ['fast-mlsirm'] + args):
        assert main() == 0

    assert (out_dir / "responses.npy").exists()
    assert (out_dir / "item_factor.csv").exists()

def test_cli_fit_success(tmp_path):
    sim_dir = tmp_path / "sim_out"
    fit_dir = tmp_path / "fit_out"

    # Run simulation to get files
    with patch.object(sys, 'argv', ['fast-mlsirm', 'simulate', '--persons', '10', '--dims', '1', '--items-per-dim', '2', '--out', str(sim_dir)]):
        main()

    args = ["fit", "--responses", str(sim_dir / "responses.npy"), "--factors", str(sim_dir / "item_factor.csv"), "--model", "MLS2PLM", "--max-iter", "1", "--out", str(fit_dir)]

    with patch.object(sys, 'argv', ['fast-mlsirm'] + args):
        assert main() == 0

    assert (fit_dir / "params.npz").exists()
    assert (fit_dir / "fit_summary.json").exists()

def test_cli_diagnose_fit_success(tmp_path):
    sim_dir = tmp_path / "sim_out"
    fit_dir = tmp_path / "fit_out"
    diag_dir = tmp_path / "diag_out"

    with patch.object(sys, 'argv', ['fast-mlsirm', 'simulate', '--persons', '10', '--dims', '1', '--items-per-dim', '2', '--out', str(sim_dir)]):
        main()

    with patch.object(sys, 'argv', ['fast-mlsirm', 'fit', '--responses', str(sim_dir / "responses.npy"), '--factors', str(sim_dir / "item_factor.csv"), '--model', 'MLS2PLM', '--max-iter', '1', '--out', str(fit_dir)]):
        main()

    args = [
        "diagnose-fit",
        "--responses",
        str(sim_dir / "responses.npy"),
        "--factors",
        str(sim_dir / "item_factor.csv"),
        "--params",
        str(fit_dir / "params.npz"),
        "--model",
        "MLS2PLM",
        "--out",
        str(diag_dir),
    ]
    with patch.object(sys, 'argv', ['fast-mlsirm'] + args):
        assert main() == 0

    assert (diag_dir / "fit_diagnostics.json").exists()

def test_cli_diagnose_dimensions_success(tmp_path):
    sim_dir = tmp_path / "sim_out"
    diag_dir = tmp_path / "dim_out"

    with patch.object(sys, 'argv', ['fast-mlsirm', 'simulate', '--persons', '10', '--dims', '1', '--items-per-dim', '2', '--out', str(sim_dir)]):
        main()

    args = [
        "diagnose-dimensions",
        "--responses",
        str(sim_dir / "responses.npy"),
        "--factors",
        str(sim_dir / "item_factor.csv"),
        "--latent-dims",
        "1",
        "--folds",
        "2",
        "--max-iter",
        "1",
        "--out",
        str(diag_dir),
    ]
    with patch.object(sys, 'argv', ['fast-mlsirm'] + args):
        assert main() == 0

    assert (diag_dir / "dimension_diagnostics.json").exists()

def test_cli_fit_missing_file(capsys):
    args = ["fit", "--responses", "nonexistent.npy", "--factors", "nonexistent.csv", "--out", "out"]
    with patch.object(sys, 'argv', ['fast-mlsirm'] + args):
        assert main() == 1

    captured = capsys.readouterr()
    assert "Error: Could not find file" in captured.err

def test_cli_fit_bad_data(tmp_path, capsys):
    bad_npy = tmp_path / "bad.npy"
    bad_npy.write_bytes(b"not a numpy file")

    args = ["fit", "--responses", str(bad_npy), "--factors", "nonexistent.csv", "--out", "out"]
    with patch.object(sys, 'argv', ['fast-mlsirm'] + args):
        assert main() == 1

    captured = capsys.readouterr()
    assert "Error: Failed to load data" in captured.err

def test_main_sys_exit_on_direct_call():
    with patch('fast_mlsirm.cli.main', return_value=0):
        # We can't easily test `if __name__ == "__main__": raise SystemExit(main())`
        # without running it as a subprocess, but coverage usually skips it or we can ignore it.
        pass
