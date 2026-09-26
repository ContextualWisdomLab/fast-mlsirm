"""Reporting metadata regressions; native numerical routines are mocked."""

import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def producer(monkeypatch):
    # Load the actual module without package __init__ or a native installation.
    # The alternate namespace prevents changing other tests' package imports.
    name = "_person_fit_validity_fixture"
    package = ModuleType(name)
    package.__path__ = [str(Path(__file__).parents[1] / "python/fast_mlsirm")]
    monkeypatch.setitem(sys.modules, name, package)
    p = importlib.import_module(f"{name}.polytomous")
    calls = []
    numerical = {
        "lz": [0.25], "lz_star": [-2.0], "theta_eap": [0.125], "flagged": [True],
    }

    def native(*args):
        calls.append(args)
        # Even a stale/incorrect core's positive metadata cannot grant validity.
        return {**numerical, "valid_person_fit": True, "diagnostic_only": False}

    core = SimpleNamespace(__file__=__file__, poly_person_fit=native)
    monkeypatch.setattr(p, "_core_module", lambda: core)
    monkeypatch.setitem(sys.modules, "fast_mlsirm", SimpleNamespace(__version__="mock-only"))
    yield p, calls, numerical
    for key in list(sys.modules):
        if key.startswith(name + "."):
            del sys.modules[key]


def fit(p, kind, converged):
    slope = np.ones(2)
    thresholds = np.array([[0.5, -0.5]] * 2)
    if kind == "fipc":
        return p.PolyFipcFit(slope, thresholds, 0.4, 1.2, 0.0, 2, converged, "fixture")
    if kind == "unknown":
        return SimpleNamespace(model="grm", slope=slope, cat_params=thresholds)
    return p.PolytomousFit("grm", slope, thresholds, 0.0, 2, converged, "fixture")


@pytest.mark.parametrize("kind,converged,override", [
    ("standard", True, False), ("fipc", True, False),
    ("standard", True, True), ("standard", False, True),
    ("unknown", None, True),
])
@pytest.mark.parametrize("alias", [False, True])
def test_no_convergence_or_override_grants_reporting_validity(producer, kind, converged, override, alias):
    p, calls, numerical = producer
    function = p.person_fit_polytomous if alias else p.compute_person_fit_polytomous
    kwargs = dict(q_theta=121, flag_threshold=-1.5, allow_unconverged=override)
    if alias:
        with pytest.warns(DeprecationWarning):
            result = function(np.array([[0, 1]]), fit(p, kind, converged), **kwargs)
    else:
        result = function(np.array([[0, 1]]), fit(p, kind, converged), **kwargs)
    assert len(calls) == 1
    assert result["converged"] == ("unknown" if kind == "unknown" else converged)
    assert result["termination_reason"] == ("unknown" if kind == "unknown" else "fixture")
    assert result["validity_schema_version"] == 1
    assert type(result["validity_schema_version"]) is int
    assert result["valid_person_fit"] is False
    assert result["diagnostic_only"] is True
    assert result["statistic_validity"] == {
        "lz": "not_assessed_uncorrected",
        "lz_star": "unverified_polytomous_eap_correction",
        "flagged": "unverified_polytomous_eap_correction",
    }
    for key, value in numerical.items():
        np.testing.assert_array_equal(result[key], value)
    assert calls[0][-1] is (kind == "fipc")


@pytest.mark.parametrize("kind,override", [("standard", False), ("unknown", False), ("fipc", False), ("fipc", True)])
def test_existing_convergence_refusal_precedes_native_call(producer, kind, override):
    p, calls, _ = producer
    with pytest.raises(ValueError, match="converged"):
        p.compute_person_fit_polytomous(
            np.array([[0, 1]]), fit(p, kind, False), q_theta=121,
            flag_threshold=-1.5, allow_unconverged=override,
        )
    assert calls == []


def test_no_caller_validity_escape_hatch(producer):
    p, calls, _ = producer
    with pytest.raises(TypeError, match="valid_person_fit"):
        p.compute_person_fit_polytomous(
            np.array([[0, 1]]), fit(p, "standard", True), q_theta=121,
            flag_threshold=-1.5, valid_person_fit=True,
        )
    assert calls == []
