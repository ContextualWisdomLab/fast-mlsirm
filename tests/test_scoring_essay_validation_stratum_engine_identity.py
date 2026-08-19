"""Regression for exact engine identity in stratified essay validation."""

from __future__ import annotations

from pathlib import Path
import runpy

import pytest

from fast_mlsirm.scoring import AssessmentSpecError

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_scoring_essay_validation_stratification.py"))
)
_automated_engine = _FIXTURES["_automated_engine"]
_report = _FIXTURES["_report"]
_stratum = _FIXTURES["_stratum"]


def _drifted_engine_subclass():
    """Return a nominal EngineDescriptor subclass with a conflicting family."""
    exact_engine = _automated_engine()

    class DriftedEngine(type(exact_engine)):
        pass

    drifted = object.__new__(DriftedEngine)
    for name, value in vars(exact_engine).items():
        object.__setattr__(drifted, name, value)
    object.__setattr__(drifted, "engine_family_id", "alternate_family")
    return exact_engine, drifted


def test_stratified_report_rejects_engine_subclass_before_identity_binding() -> None:
    """A subclass must not bypass the stratum-to-engine family consistency gate."""
    exact_engine, drifted_engine = _drifted_engine_subclass()

    with pytest.raises(AssessmentSpecError) as caught:
        _report(
            stratum=_stratum(engine=exact_engine),
            engine=drifted_engine,
        )

    assert caught.value.code == "invalid_essay_validation_automated_engine"
