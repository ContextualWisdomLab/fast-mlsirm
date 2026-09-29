"""Receipt contract for the S1 two-tier GRM FIPC consumer verification script."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads(
    (ROOT / "contracts" / "s1-fipc-consumer-verify-v1.schema.json").read_text()
)


def _load_script():
    script = ROOT / "scripts" / "s1_fipc_consumer_verify.py"
    spec = importlib.util.spec_from_file_location("s1_fipc_consumer_verify", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _assert_schema(receipt: dict) -> None:
    properties = SCHEMA["properties"]
    assert set(SCHEMA["required"]) <= receipt.keys()
    assert receipt.keys() <= properties.keys()
    assert receipt["row_order_diagnosis"] in properties["row_order_diagnosis"]["enum"]


def _fake_core(module):
    """A binding stand-in that returns flattened arrays, as the PyO3 layer does."""
    m = module

    def fit(*_args, **_kwargs):
        return {
            "a_primary": m.TRUE_A_P.reshape(-1).copy(),
            "a_specific": m.TRUE_A_S.copy(),
            "threshold": m.TRUE_D.reshape(-1).copy(),
        }

    def fipc(y, *args):
        fixed_a_p, fixed_a_s, fixed_d, max_iter = args[9], args[10], args[11], args[14]
        n_persons = y.size // m.N_ITEMS
        # Row-equivariant stand-in: each person's EAP depends only on their own row.
        theta = np.column_stack((y.reshape(n_persons, m.N_ITEMS)[:, :3].sum(axis=1),
                                 y.reshape(n_persons, m.N_ITEMS)[:, 3:].sum(axis=1))) * 0.1
        converged = max_iter > 1
        return {
            "converged": converged,
            "termination_reason": "tolerance" if converged else "max_iter_reached",
            "theta_p_eap": theta.reshape(-1),
            "a_primary": np.asarray(fixed_a_p).copy(),
            "a_specific": np.asarray(fixed_a_s).copy(),
            "threshold": np.asarray(fixed_d).copy(),
            "primary_mean": np.array([0.6, -0.3]),
            "primary_sd": np.array([1.2, 0.85]),
            "specific_sd": np.ones(m.N_SPECIFIC),
        }

    return SimpleNamespace(__file__=m._core.__file__, fit_two_tier_grm=fit,
                           fit_two_tier_grm_fipc=fipc)


def test_receipt_reports_missing_binding_honestly(monkeypatch) -> None:
    """Without the FIPC binding the receipt says so instead of hiding a TypeError."""
    module = _load_script()
    monkeypatch.delattr(module._core, "fit_two_tier_grm_fipc", raising=False)
    receipt = module.build_receipt(consumer_sha=None, build_source_sha=None)

    _assert_schema(receipt)
    assert receipt["row_order_diagnosis"] == "binding_unavailable"
    assert receipt["anchor_identified"] is True
    assert receipt["row_order_max_abs"] is None
    assert receipt["build_source_sha_authority"] == "caller_asserted"
    assert receipt["all_pass"] is False
    assert "fipc_fit_error" not in receipt


def test_rank_deficient_anchor_fails_closed_before_fitting(monkeypatch) -> None:
    """Anchors loading on only one primary leave the other focal mean unidentified."""
    module = _load_script()
    monkeypatch.setattr(module, "ANCHOR", np.array([1, 1, 1, 0, 0, 0], dtype=bool))
    monkeypatch.setattr(module, "_core", _fake_core(module))
    receipt = module.build_receipt(consumer_sha=None, build_source_sha=None)

    _assert_schema(receipt)
    assert receipt["anchored_primary_rank"] == 1
    assert receipt["anchor_identified"] is False
    assert receipt["row_order_diagnosis"] == "not_identified"
    assert receipt["row_order_max_abs"] is None
    assert receipt["all_pass"] is False


def test_specific_variance_estimation_needs_an_anchor_per_specific_block() -> None:
    module = _load_script()
    anchor = np.array([1, 0, 0, 1, 0, 0], dtype=bool)  # spans both primaries, only specific 0
    assert module.anchor_identification(module.PRIMARY_MAP, module.SPECIFIC_MAP, anchor, False)["anchor_identified"]
    assert not module.anchor_identification(module.PRIMARY_MAP, module.SPECIFIC_MAP, anchor, True)["anchor_identified"]


def test_flattened_binding_outputs_are_reshaped_per_person(monkeypatch) -> None:
    """theta_p_eap arrives as n_persons*n_primary; the gates must work per person."""
    module = _load_script()
    monkeypatch.setattr(module, "_core", _fake_core(module))
    receipt = module.build_receipt(consumer_sha=None, build_source_sha=None)

    _assert_schema(receipt)
    assert "fipc_fit_error" not in receipt, receipt.get("fipc_fit_error")
    assert receipt["row_order_diagnosis"] == "pass"
    assert receipt["row_order_max_abs"] == 0.0
    assert receipt["anchor_rows_fixed"] is True
    assert receipt["responses_fit_eap_expected_raw"] is True
    assert receipt["focal_primary_mean_error"] == pytest.approx([0.6 - 0.65, -0.3 + 0.35])
