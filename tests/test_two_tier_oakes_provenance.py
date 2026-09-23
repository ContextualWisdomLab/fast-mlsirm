"""Identification forwarding contracts; no fitting or native computation."""

import ast
import importlib
import inspect
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def adapter(monkeypatch):
    name = "_two_tier_oakes_provenance_fixture"
    package = ModuleType(name)
    package.__path__ = [str(Path(__file__).parents[1] / "python/fast_mlsirm")]
    monkeypatch.setitem(sys.modules, name, package)
    calls = []

    def native(*args):
        calls.append(args)
        # Sentinels only: the test verifies transport, not numerical accuracy.
        return dict(labels=["sentinel"], information=[4.0], vcov=None, se=None,
                    positive_definite=False, non_pd_reason="sentinel")

    fitstats = ModuleType(f"{name}.fitstats")
    fitstats._core_module = lambda: SimpleNamespace(two_tier_oakes_se=native)
    monkeypatch.setitem(sys.modules, fitstats.__name__, fitstats)
    module = importlib.import_module(f"{name}.two_tier_grm")
    yield module, calls
    for key in list(sys.modules):
        if key.startswith(name + "."):
            del sys.modules[key]


def inputs():
    return dict(
        a_primary=np.eye(2), a_specific=np.ones(2),
        threshold=np.array([[0.5, -0.5], [0.5, -0.5]]), phi=np.eye(2),
        responses=np.array([[0, 1]]), primary_map=np.eye(2, dtype=bool),
        specific_map=np.zeros(2, dtype=int), n_cat=3, n_primary=2, n_specific=1,
        q_primary=121, q_specific=121, fd_step=1e-5,
    )


def fit_inputs(record):
    data = inputs()
    fields = {key: data.pop(key) for key in (
        "a_primary", "a_specific", "threshold", "phi", "n_cat", "n_primary", "n_specific"
    )}
    return SimpleNamespace(**fields, primary_identification=record), data


@pytest.mark.parametrize("record,mode", [("orthogonal", "identity"), ("correlated", "estimate")])
def test_same_identity_matrix_keeps_estimator_record(adapter, record, mode):
    module, calls = adapter
    fit, data = fit_inputs(record)
    result = module.two_tier_oakes_se_from_fit(fit, **data)
    assert len(calls) == 1
    assert calls[0][-1] == mode
    np.testing.assert_array_equal(calls[0][3], np.eye(2).reshape(-1))
    assert result.primary_correlation == mode
    assert result.identification_source == "fit.primary_identification"
    assert result.information.tolist() == [[4.0]]
    assert result.se is None and result.non_pd_reason == "sentinel"


@pytest.mark.parametrize("record", [None, "", "unknown", "identity", True, []])
def test_unknown_record_refuses_before_native(adapter, record):
    module, calls = adapter
    fit, data = fit_inputs(record)
    with pytest.raises(ValueError, match="primary_identification"):
        module.two_tier_oakes_se_from_fit(fit, **data)
    assert calls == []


def test_old_checkpoint_missing_record_refuses(adapter):
    module, calls = adapter
    fit, data = fit_inputs("orthogonal")
    del fit.primary_identification
    with pytest.raises(ValueError, match="missing provenance"):
        module.two_tier_oakes_se_from_fit(fit, **data)
    assert calls == []


def test_record_cannot_be_overridden(adapter):
    module, calls = adapter
    fit, data = fit_inputs("orthogonal")
    with pytest.raises(TypeError, match="primary_correlation"):
        module.two_tier_oakes_se_from_fit(fit, **data, primary_correlation="estimate")
    assert calls == []


@pytest.mark.parametrize("value", [0.1, np.nan, np.inf])
def test_contradictory_orthogonal_matrix_refuses_before_native(adapter, value):
    module, calls = adapter
    fit, data = fit_inputs("orthogonal")
    fit.phi[0, 1] = value
    with pytest.raises(ValueError, match="requires phi = I"):
        module.two_tier_oakes_se_from_fit(fit, **data)
    assert calls == []


@pytest.mark.parametrize("mode", ["identity", "estimate"])
def test_explicit_array_mode_is_distinct_from_fit_provenance(adapter, mode):
    module, calls = adapter
    result = module.two_tier_oakes_se(**inputs(), primary_correlation=mode)
    assert calls[0][-1] == mode
    assert result.primary_correlation == mode
    assert result.identification_source == "explicit_array_mode"


def test_array_omission_and_invalid_mode_refuse(adapter):
    module, calls = adapter
    with pytest.raises(TypeError, match="primary_correlation"):
        module.two_tier_oakes_se(**inputs())
    for mode in (None, "orthogonal", True):
        with pytest.raises(ValueError, match="primary_correlation"):
            module.two_tier_oakes_se(**inputs(), primary_correlation=mode)
    assert calls == []
    assert inspect.signature(module.fit_two_tier_grm).parameters["primary_correlation"].default == "estimate"


def test_public_export_and_native_signature_source_contract():
    root = Path(__file__).parents[1]
    tree = ast.parse((root / "python/fast_mlsirm/_legacy_init.py").read_text())
    name = "two_tier_oakes_se_from_fit"
    assert any(isinstance(node, ast.ImportFrom) and node.module == "two_tier_grm"
               and any(alias.name == name for alias in node.names) for node in tree.body)
    exports = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets))
    assert name in ast.literal_eval(exports)
    native = (root / "crates/fast-mlsirm-py/src/lib.rs").read_text()
    prefix = native[:native.index("fn two_tier_oakes_se(")]
    signature = prefix[prefix.rindex("#[pyo3(signature"):]
    assert 'primary_correlation = "estimate"' not in signature
    assert "primary_correlation\n" in signature
    # This source assertion is not a substitute for compiling the new binding.
