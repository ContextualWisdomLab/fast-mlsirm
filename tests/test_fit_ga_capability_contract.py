"""GA capability-contract regressions for the public fit entry point."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from fast_mlsirm import FitConfig


_FIT_PATH = Path(__file__).parents[1] / "python" / "fast_mlsirm" / "fit.py"


def _public_fit_node() -> ast.FunctionDef:
    module = ast.parse(_FIT_PATH.read_text(encoding="utf-8"))
    node = next(
        candidate
        for candidate in module.body
        if isinstance(candidate, ast.FunctionDef) and candidate.name == "fit"
    )
    return node


def test_public_fit_has_no_not_implemented_error_surface() -> None:
    """GA fit capability rejection must stay in validated public configuration."""
    fit_node = _public_fit_node()
    raises_not_implemented = [
        node
        for node in ast.walk(fit_node)
        if isinstance(node, ast.Raise)
        and isinstance(node.exc, ast.Call)
        and isinstance(node.exc.func, ast.Name)
        and node.exc.func.id == "NotImplementedError"
    ]
    assert not raises_not_implemented


@pytest.mark.parametrize("estimator", ["em", "bayes"])
def test_reserved_estimators_fail_as_configuration_errors(estimator: str) -> None:
    with pytest.raises(ValueError, match="estimator must be one of"):
        FitConfig(estimator=estimator)


def test_bifactor_jmle_fails_as_configuration_error() -> None:
    with pytest.raises(ValueError, match="BIFAC2PLM requires estimator 'mmle'"):
        FitConfig(model="BIFAC2PLM", estimator="jmle")
