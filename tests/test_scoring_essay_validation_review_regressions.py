"""Regression tests for reviewed essay validation evidence behaviors."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import runpy

import pytest

import fast_mlsirm.scoring.essay.validation_reporting as validation_reporting
from fast_mlsirm.validation import validate_judge

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_scoring_essay_validation_reporting.py"))
)
_build_report = _FIXTURES["build_report"]
_AUTOMATED = _FIXTURES["_AUTOMATED"]
_REFERENCE = _FIXTURES["_REFERENCE"]
_HUMAN_A = _FIXTURES["_HUMAN_A"]
_HUMAN_B = _FIXTURES["_HUMAN_B"]
_SUBGROUP = _FIXTURES["_SUBGROUP"]

_GATE_METRIC_IDS = {
    "qwk": "quadratic_weighted_kappa",
    "pearson_r": "pearson_correlation",
    "smd": "standardized_mean_difference",
    "degradation": "human_machine_degradation",
    "subgroup_smd": "worst_subgroup_standardized_mean_difference",
}


def test_validation_metric_mapping_is_gate_order_invariant(monkeypatch) -> None:
    """Gate reordering must not change the metric identity assigned to a value."""
    original = validate_judge(
        _AUTOMATED,
        _REFERENCE,
        k=3,
        human_human=(_HUMAN_A, _HUMAN_B),
        subgroup=_SUBGROUP,
    )
    reordered = replace(original, gates=list(reversed(original.gates)))
    monkeypatch.setattr(
        validation_reporting,
        "validate_judge",
        lambda *args, **kwargs: reordered,
    )

    report = _build_report()
    expected_values = {
        _GATE_METRIC_IDS[gate["name"]]: gate["value"] for gate in reordered.gates
    }
    expected_values["exact_agreement"] = reordered.exact_agreement
    expected_values["adjacent_agreement"] = reordered.adjacent_agreement

    assert {metric.metric_id: metric.value for metric in report.metrics} == pytest.approx(
        expected_values
    )


def test_validation_report_serialization_omits_all_label_vectors() -> None:
    """No source label vector may appear anywhere in the nested report payload."""
    serialized_payload = json.dumps(_build_report().to_dict(), sort_keys=True)

    for label_vector in (
        _AUTOMATED,
        _REFERENCE,
        _HUMAN_A,
        _HUMAN_B,
        _SUBGROUP,
    ):
        assert json.dumps(label_vector.tolist()) not in serialized_payload
