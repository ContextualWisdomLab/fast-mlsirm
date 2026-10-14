"""Caller-control integration for the calibration CLI, without native fitting."""

import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fast_mlsirm import fit_bifactor_grm_multigroup


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "bifactor_multigroup_oakes_calibration.py"
    spec = importlib.util.spec_from_file_location("calibration_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _argv():
    return ["calibration", "--replicates", "3", "--persons-per-group", "2",
            "--q-general", "121", "--q-specific", "241", "--fd-step", "0.00001"]


def test_calibration_forwards_explicit_estep_controls(monkeypatch):
    module = _module()
    calls = []
    signature = inspect.signature(fit_bifactor_grm_multigroup)

    def recording_fit(*args, **kwargs):
        signature.bind(*args, **kwargs)
        calls.append(kwargs.copy())
        return SimpleNamespace(converged=True, a_general=np.ones((3, 13)))

    monkeypatch.setattr(module, "fit_bifactor_grm_multigroup", recording_fit)
    monkeypatch.setattr(module, "bifactor_multigroup_oakes_se", lambda *a, **k:
                        SimpleNamespace(positive_definite=True, labels=["a_general:0"], se=np.ones(1)))
    monkeypatch.setattr("sys.argv", _argv() + ["--e-step-n-chunks", "3", "--e-step-n-threads", "2"])
    module.main()
    assert len(calls) == 3
    for rep, call in enumerate(calls):
        assert call["e_step_n_chunks"] == 3
        assert call["e_step_n_threads"] == 2
        assert call["q_general"] == 121
        assert call["q_specific"] == 241
        assert call["seed"] == 2113 + rep
        assert call["max_iter"] == 500
        assert call["tol"] == 1e-6


def test_precision_only_preserves_no_fit_path_without_estep_controls(monkeypatch):
    module = _module()
    calls = []

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("precision-only must not fit a model")

    def recording_oakes(*args, **kwargs):
        calls.append(kwargs.copy())
        return SimpleNamespace(positive_definite=True, non_pd_reason=None, se=np.ones(1))

    monkeypatch.setattr(module, "fit_bifactor_grm_multigroup", forbidden_fit)
    monkeypatch.setattr(module, "bifactor_multigroup_oakes_se", recording_oakes)
    monkeypatch.setattr("sys.argv", _argv() + ["--precision-only", "--compare-q-general", "241", "--compare-q-specific", "481"])
    module.main()
    assert len(calls) == 2
    assert [(c["q_general"], c["q_specific"]) for c in calls] == [(121, 241), (241, 481)]
    assert all(c["fd_step"] == 0.00001 for c in calls)


@pytest.mark.parametrize("omitted", ["chunks", "threads"])
def test_calibration_requires_each_estep_control_before_fit(monkeypatch, capsys, omitted):
    module = _module()
    def forbidden_fit(*args, **kwargs):
        raise AssertionError("fit executed before CLI control admission")
    monkeypatch.setattr(module, "fit_bifactor_grm_multigroup", forbidden_fit)
    remaining = ["--e-step-n-threads", "2"] if omitted == "chunks" else ["--e-step-n-chunks", "3"]
    monkeypatch.setattr("sys.argv", _argv() + remaining)
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2
    message = capsys.readouterr().err
    assert "the following arguments are required: --e-step-n-" + omitted in message
