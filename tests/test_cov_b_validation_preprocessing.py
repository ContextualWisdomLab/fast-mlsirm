"""Coverage-B: guard branches of validation.py and preprocessing.py."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm import preprocessing, validation
from fast_mlsirm.validation import _validate_labels


# -- validation --------------------------------------------------------------


def test_validate_labels_rejects_empty_array():
    with pytest.raises(ValueError, match="must be non-empty"):
        _validate_labels(np.array([]), "judge")


def test_validate_judge_rejects_too_few_categories():
    with pytest.raises(ValueError, match="must be >= 2"):
        validation.validate_judge(np.array([0, 1]), np.array([0, 1]), k=1)


def test_validate_judge_accepts_human_human_baseline():
    judge = np.array([0, 1, 1, 0, 1, 0])
    human = np.array([0, 1, 0, 0, 1, 1])
    verdict = validation.validate_judge(
        judge, human, k=2, human_human=(np.array([0, 1, 0, 0, 1, 0]), human)
    )
    assert isinstance(verdict, validation.ValidationVerdict)
    assert isinstance(verdict.passed, bool)
    assert any(g["name"] == "degradation" for g in verdict.gates)


def test_validation_policy_rejects_empty_identity_and_invalid_thresholds():
    """Policy metadata and every threshold remain bounded before Rust dispatch."""
    for field, value, message in (
        ("policy_id", "", "policy_id"),
        ("policy_version", "", "policy_version"),
        ("qwk_min", "bad", "qwk_min"),
        ("min_subgroup_n", 1, "min_subgroup_n"),
    ):
        with pytest.raises(ValueError, match=message):
            validation.ValidationPolicy(**{field: value})


def test_validate_judge_rejects_an_untyped_policy():
    """The public gate requires a validated ValidationPolicy instance."""
    with pytest.raises(TypeError, match="policy must be a ValidationPolicy"):
        validation.validate_judge(np.array([0, 1]), np.array([0, 1]), policy=object())


def test_agreement_validates_object_float_and_text_rating_payloads():
    """Fleiss and Light adapters reject payloads that could truncate or wrap."""
    fleiss = validation.fleiss_kappa(
        np.array([[0, 1, 0], [1, 0, 1]], dtype=object),
        k=2,
    )
    assert isinstance(fleiss, validation.FleissKappaResult)
    with pytest.raises(ValueError, match="dtype .* is not numeric"):
        validation.fleiss_kappa(np.array([["a", "b"]]))
    with pytest.raises(ValueError, match="exact float64 integer range"):
        validation.fleiss_kappa(np.array([[2.0**53 + 2, 0.0]]), k=2)

    light = validation.light_kappa(np.array([[0, 1], [1, 0]], dtype=np.int64))
    assert isinstance(light, validation.LightKappaResult)
    with pytest.raises(ValueError, match="dtype .* is not numeric"):
        validation.light_kappa(np.array([["a", "b"]]))
    with pytest.raises(ValueError, match=r"<= 2\^32"):
        validation.light_kappa(np.array([[2.0**32 + 1, 0.0]]))


# -- preprocessing -----------------------------------------------------------


def test_irtree_expand_rejects_non_2d_mapping():
    responses = np.array([[0.0], [1.0]])
    with pytest.raises(ValueError, match="mapping must be nodes x categories"):
        preprocessing.irtree_expand(responses, np.array([0.0, 1.0]))


def test_irtree_expand_handles_all_missing_responses():
    responses = np.array([[np.nan], [np.nan]])
    mapping = np.array([[0.0, 1.0]])
    expanded, factor_id = preprocessing.irtree_expand(responses, mapping)
    assert expanded.shape == (2, 1)
    assert np.all(np.isnan(expanded))
    assert np.array_equal(factor_id, np.array([0]))


def test_irtree_expand_rejects_wrong_shape_node_dims():
    responses = np.array([[0.0], [1.0]])
    mapping = np.array([[0.0, 1.0]])
    with pytest.raises(ValueError, match="node_dims must have one entry"):
        preprocessing.irtree_expand(responses, mapping, node_dims=np.array([0, 1]))


def test_irtree_expand_rejects_non_numeric_node_dims():
    responses = np.array([[0.0], [1.0]])
    mapping = np.array([[0.0, 1.0]])
    with pytest.raises(ValueError, match="node_dims must be finite non-negative integers"):
        preprocessing.irtree_expand(responses, mapping, node_dims=np.array(["x"]))
