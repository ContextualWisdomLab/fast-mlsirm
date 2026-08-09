"""Fail-first contract for the governed factor-retention evidence surface."""

from __future__ import annotations

import importlib.util


def test_factor_retention_contract_module_exists() -> None:
    """Issue #608 requires a dedicated factor-retention contract namespace."""
    assert importlib.util.find_spec("fast_mlsirm.factor_retention") is not None
