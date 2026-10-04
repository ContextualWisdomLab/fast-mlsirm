"""Hybrid scoring documentation contracts; source only, no native execution.

Cai (2015, pp. 542-543, Eqs. 14-17) identifies conditional scoring;
Higham (1993, pp. 785-786, Eqs. 2.6-2.8) separates accumulator rounding.
These lexical checks do not establish semantic or scientific acceptance.

References:
Cai, L. (2015). Lord-Wingersky algorithm version 2.0 for hierarchical item
factor models with applications in test scoring, scale alignment, and model
fit testing. Psychometrika, 80(2), 535-559. doi:10.1007/s11336-014-9411-3.
Higham, N. J. (1993). The accuracy of floating point summation. SIAM Journal
on Scientific Computing, 14(4), 783-799. doi:10.1137/0914050.
"""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("kind", ["python", "dispatch", "binding", "changelog"])
def test_hybrid_scoring_contract_description(kind):
    if kind == "python":
        tree = ast.parse((ROOT / "python/fast_mlsirm/two_tier_grm.py").read_bytes())
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef)
                        and node.name == "expected_raw_two_tier_grm")
        docs = ast.get_docstring(function)
        assert docs is not None
    elif kind in ("dispatch", "binding"):
        path, start, end = {
            "dispatch": ("crates/mlsirm-core/src/two_tier_recursion.rs",
                         "/// Device-dispatched", "pub fn two_tier_expected_raw_on("),
            "binding": ("crates/fast-mlsirm-py/src/lib.rs",
                        "/// Plug-in expected raw total scores", "#[pyfunction]"),
        }[kind]
        source = (ROOT / path).read_text()
        first = source.index(start)
        last = source.index(end, first)
        docs = source[first:last].replace("///", "")
    else:
        docs = (ROOT / "docs/changelog.d/two-tier-expected-raw-gpu.md").read_text()
    docs = " ".join(docs.split())
    assert "f32 contributions" in docs and "host f64" in docs
    assert "not all-GPU" in docs
    assert "compensated summation" not in docs
    assert "rows * items * nodes * 4" in docs
    assert "caller" in docs and "node count" in docs
    if kind != "changelog":
        assert "Higham (1993, pp. 785-786, Eqs. 2.6-2.8)" in docs
        assert "Higham, N. J. (1993)." in docs
        assert "Cai" in docs and "542" in docs and "543" in docs
    if kind in ("python", "binding"):
        assert "does not expose" in docs and "used_gpu" in docs
