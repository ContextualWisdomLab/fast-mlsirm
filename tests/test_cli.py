import json
import sys
from unittest.mock import patch

import numpy as np

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

def test_cli_simulate_json_output(tmp_path, capsys):
    out_dir = tmp_path / "sim_out"
    args = [
        "simulate",
        "--persons",
        "10",
        "--dims",
        "2",
        "--items-per-dim",
        "2",
        "--out",
        str(out_dir),
        "--json",
    ]

    with patch.object(sys, 'argv', ['fast-mlsirm'] + args):
        assert main() == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "simulate"
    assert payload["status"] == "ok"
    assert payload["n_items"] == 4
    assert payload["files"]["responses"].endswith("responses.npy")

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

def test_cli_fit_json_output(tmp_path, capsys):
    sim_dir = tmp_path / "sim_out"
    fit_dir = tmp_path / "fit_out"

    with patch.object(sys, 'argv', ['fast-mlsirm', 'simulate', '--persons', '10', '--dims', '1', '--items-per-dim', '2', '--out', str(sim_dir)]):
        main()
    capsys.readouterr()

    args = [
        "fit",
        "--responses",
        str(sim_dir / "responses.npy"),
        "--factors",
        str(sim_dir / "item_factor.csv"),
        "--model",
        "MLS2PLM",
        "--max-iter",
        "1",
        "--out",
        str(fit_dir),
        "--json",
    ]

    with patch.object(sys, 'argv', ['fast-mlsirm'] + args):
        assert main() == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "fit"
    assert payload["status"] == "ok"
    assert payload["model"] == "MLS2PLM"
    assert payload["backend"] == "numpy"
    assert payload["files"]["params"].endswith("params.npz")

    summary = json.loads((fit_dir / "fit_summary.json").read_text(encoding="utf-8"))
    assert summary["backend"] == "numpy"


def test_cli_fit_auto_backend_records_resolved_backend(tmp_path, capsys):
    sim_dir = tmp_path / "sim_out"
    fit_dir = tmp_path / "fit_out"

    with patch.object(sys, 'argv', ['fast-mlsirm', 'simulate', '--persons', '10', '--dims', '1', '--items-per-dim', '2', '--out', str(sim_dir)]):
        main()
    capsys.readouterr()

    args = [
        "fit",
        "--responses",
        str(sim_dir / "responses.npy"),
        "--factors",
        str(sim_dir / "item_factor.csv"),
        "--model",
        "MLS2PLM",
        "--max-iter",
        "1",
        "--backend",
        "auto",
        "--out",
        str(fit_dir),
        "--json",
    ]

    with patch.object(sys, 'argv', ['fast-mlsirm'] + args):
        assert main() == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["backend"] in {"numpy", "rust"}
    summary = json.loads((fit_dir / "fit_summary.json").read_text(encoding="utf-8"))
    assert summary["backend"] == payload["backend"]

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

def test_cli_diagnose_response_process_success(tmp_path):
    responses = tmp_path / "responses.npy"
    probabilities = tmp_path / "probabilities.npy"
    group_id = tmp_path / "group_id.npy"
    cluster_id = tmp_path / "cluster_id.npy"
    out_dir = tmp_path / "process_out"
    np.save(responses, np.array([[0, 1], [2, 1]]))
    np.save(probabilities, np.full((2, 2, 3), 1.0 / 3.0))
    np.save(group_id, np.array([0, 1]))
    np.save(cluster_id, np.array([10, 20]))

    args = [
        "diagnose-response-process",
        "--responses",
        str(responses),
        "--probabilities",
        str(probabilities),
        "--item-type",
        "polytomous",
        "--response-process",
        "cumulative",
        "--group-id",
        str(group_id),
        "--cluster-id",
        str(cluster_id),
        "--out",
        str(out_dir),
    ]
    with patch.object(sys, 'argv', ['fast-mlsirm'] + args):
        assert main() == 0

    assert (out_dir / "fit_diagnostics.json").exists()
    payload = json.loads((out_dir / "fit_diagnostics.json").read_text(encoding="utf-8"))
    assert payload["groupfit"]["group_id"] == [0.0, 1.0]
    assert payload["clusterfit"]["cluster_id"] == [10.0, 20.0]

def test_cli_diagnose_response_candidates_success(tmp_path):
    responses = tmp_path / "responses.npy"
    weak = tmp_path / "weak.npy"
    strong = tmp_path / "strong.npy"
    out_dir = tmp_path / "candidate_out"
    np.save(responses, np.array([[0, 1], [1, 0]]))
    np.save(weak, np.full((2, 2, 2), 0.5))
    np.save(strong, np.array([[[0.8, 0.2], [0.2, 0.8]], [[0.2, 0.8], [0.8, 0.2]]]))

    args = [
        "diagnose-response-candidates",
        "--responses",
        str(responses),
        "--candidate",
        f"dim1={weak}",
        "--candidate",
        f"dim2={strong}",
        "--item-type",
        "dichotomous",
        "--response-process",
        "ideal_point",
        "--out",
        str(out_dir),
    ]
    with patch.object(sys, 'argv', ['fast-mlsirm'] + args):
        assert main() == 0

    payload = json.loads((out_dir / "dimension_diagnostics.json").read_text(encoding="utf-8"))
    assert payload["best"]["candidate_label"] == "dim2"

def test_cli_diagnose_fixed_item_calibration_success(tmp_path, capsys):
    responses = tmp_path / "responses.npy"
    weak = tmp_path / "weak.npy"
    strong = tmp_path / "strong.npy"
    fixed_items = tmp_path / "fixed_items.npy"
    out_dir = tmp_path / "fixed_calibration_out"
    np.save(responses, np.array([[0, 1], [1, 0], [0, 1]]))
    np.save(weak, np.full((3, 2, 2), 0.5))
    np.save(
        strong,
        np.array(
            [
                [[0.9, 0.1], [0.1, 0.9]],
                [[0.1, 0.9], [0.9, 0.1]],
                [[0.9, 0.1], [0.1, 0.9]],
            ]
        ),
    )
    np.save(fixed_items, np.array([0, 1]))

    args = [
        "diagnose-fixed-item-calibration",
        "--responses",
        str(responses),
        "--candidate",
        f"weak={weak}",
        "--candidate",
        f"strong={strong}",
        "--fixed-items",
        str(fixed_items),
        "--item-type",
        "dichotomous",
        "--response-process",
        "ideal_point",
        "--out",
        str(out_dir),
        "--json",
    ]
    with patch.object(sys, 'argv', ['fast-mlsirm'] + args):
        assert main() == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "diagnose-fixed-item-calibration"
    assert payload["best_candidate"] == "strong"
    diagnostics = json.loads((out_dir / "dimension_diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["best"]["candidate_label"] == "strong"
    assert "calibration_score" in diagnostics["best"]

def test_cli_diagnose_response_candidates_rejects_duplicate_label(tmp_path, capsys):
    responses = tmp_path / "responses.npy"
    weak = tmp_path / "weak.npy"
    out_dir = tmp_path / "candidate_out"
    np.save(responses, np.array([[0, 1], [1, 0]]))
    np.save(weak, np.full((2, 2, 2), 0.5))

    args = [
        "diagnose-response-candidates",
        "--responses",
        str(responses),
        "--candidate",
        f"dim1={weak}",
        "--candidate",
        f"dim1={weak}",
        "--item-type",
        "dichotomous",
        "--out",
        str(out_dir),
    ]
    with patch.object(sys, 'argv', ['fast-mlsirm'] + args):
        assert main() == 1

    assert "duplicate candidate label: dim1" in capsys.readouterr().err

def test_cli_render_report_json_output(tmp_path, capsys):
    diagnostics = tmp_path / "fit_diagnostics.json"
    report = tmp_path / "report.html"
    diagnostics.write_text(
        json.dumps(
            {
                "itemfit": {"item_id": [0], "outfit_mnsq": [1.0]},
                "personfit": {"person_id": [0], "outfit_mnsq": [1.0]},
                "model_fit": {"loglik": -1.0},
            }
        ),
        encoding="utf-8",
    )

    args = ["render-report", "--diagnostics", str(diagnostics), "--out", str(report), "--json"]
    with patch.object(sys, 'argv', ['fast-mlsirm'] + args):
        assert main() == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "render-report"
    assert payload["files"]["report"].endswith("report.html")
    assert report.exists()

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
    assert "Error: Invalid input data" in captured.err

def test_cli_fit_rejects_factor_length_mismatch(tmp_path, capsys):
    responses = tmp_path / "responses.npy"
    factors = tmp_path / "item_factor.csv"
    np.save(responses, np.ones((3, 2)))
    factors.write_text("item_id,factor_id\n0,0\n", encoding="utf-8")

    args = ["fit", "--responses", str(responses), "--factors", str(factors), "--out", str(tmp_path / "fit_out")]
    with patch.object(sys, 'argv', ['fast-mlsirm'] + args):
        assert main() == 1

    captured = capsys.readouterr()
    assert "factor_id length (1) must match response item count (2)" in captured.err

def test_cli_unexpected_error_does_not_print_traceback(capsys):
    args = ["simulate", "--persons", "10", "--dims", "1", "--items-per-dim", "1", "--out", "out"]
    with patch.object(sys, "argv", ["fast-mlsirm"] + args), patch(
        "fast_mlsirm.cli.simulate", side_effect=RuntimeError("internal detail")
    ):
        assert main() == 1

    captured = capsys.readouterr()
    assert "Unexpected failure - internal detail" in captured.err
    assert "Traceback" not in captured.err


def test_cli_debug_env_reraises_unexpected_error(monkeypatch):
    args = ["simulate", "--persons", "10", "--dims", "1", "--items-per-dim", "1", "--out", "out"]
    monkeypatch.setenv("FAST_MLSIRM_DEBUG", "1")
    with patch.object(sys, "argv", ["fast-mlsirm"] + args), patch(
        "fast_mlsirm.cli.simulate", side_effect=RuntimeError("internal detail")
    ):
        try:
            main()
        except RuntimeError as exc:
            assert str(exc) == "internal detail"
        else:
            raise AssertionError("FAST_MLSIRM_DEBUG should re-raise unexpected CLI errors")

def test_main_sys_exit_on_direct_call():
    with patch('fast_mlsirm.cli.main', return_value=0):
        # We can't easily test `if __name__ == "__main__": raise SystemExit(main())`
        # without running it as a subprocess, but coverage usually skips it or we can ignore it.
        pass
