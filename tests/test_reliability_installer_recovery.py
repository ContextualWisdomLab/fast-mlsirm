"""Regression tests for interrupted reliability-adapter installation recovery."""

from __future__ import annotations

from types import ModuleType

from fast_mlsirm import _icc_control_safety as icc_control_safety
from fast_mlsirm import reliability

_RATER_APIS = (
    "kripp_alpha",
    "finn_coefficient",
    "maxwell_re",
    "robinson_a",
)


def test_icc_installer_repairs_missing_rater_adapters_after_partial_install():
    """A hardened ICC surface must not prevent repair of sibling rater adapters."""
    module = ModuleType("fast_mlsirm_test_reliability")
    module.__package__ = "fast_mlsirm_test"
    module.icc = reliability.icc

    assert getattr(module.icc, "__fast_mlsirm_icc_control_hardened__", False)

    originals = {}
    for name in _RATER_APIS:
        wrapped = getattr(reliability, name)
        original = wrapped.__wrapped__
        originals[name] = original
        setattr(module, name, original)

    assert not getattr(
        module.kripp_alpha,
        "__fast_mlsirm_rater_evidence_hardened__",
        False,
    )

    icc_control_safety.install(module)

    assert getattr(
        module.kripp_alpha,
        "__fast_mlsirm_rater_evidence_hardened__",
        False,
    )
    for name, original in originals.items():
        assert getattr(module, name) is not original
