"""Direct boundary coverage for scoring canonicalization primitives."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum

import pytest

from fast_mlsirm.scoring import AssessmentSpecError, artifact_digest, canonical_json
import fast_mlsirm.scoring._validation as validation


class _SampleEnum(Enum):
    """Small enum used to exercise the generic enum validator."""

    ENABLED = "enabled"


class _BadItemsMapping(Mapping):
    """Mapping whose items method cannot be inspected safely."""

    def __getitem__(self, key):
        return "value"

    def __iter__(self):
        return iter(("key",))

    def __len__(self):
        return 1

    def items(self):
        raise TypeError("items unavailable")


class _MalformedItemsMapping(Mapping):
    """Mapping-like value with a malformed item tuple."""

    def __getitem__(self, key):
        return "value"

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0

    def items(self):
        return iter((("key",),))


class _ExplodingItemsIterator:
    """Iterator that raises while a mapping is being materialized."""

    def __init__(self) -> None:
        self._first = True

    def __iter__(self):
        return self

    def __next__(self):
        if self._first:
            self._first = False
            return "first", 1
        raise ValueError("iterator failure")


class _PartiallyBrokenMapping(Mapping):
    """Mapping-like value whose item iterator fails after one valid entry."""

    def __getitem__(self, key):
        return 1

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0

    def items(self):
        return _ExplodingItemsIterator()


class _DuplicateItemsMapping(Mapping):
    """Mapping-like value with two entries sharing one metadata key."""

    def __getitem__(self, key):
        return 1

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0

    def items(self):
        return iter((("same_key", 1), ("same_key", 2)))


class _CanonicalContract(validation.CanonicalContract):
    """Package-owned canonical value for the contract branch."""

    def _content_dict(self):
        return {"contract_value": "stable"}


def _error(action):
    """Return the structured error raised by one validation action."""
    with pytest.raises(AssessmentSpecError) as captured:
        action()
    return captured.value


def test_scalar_and_error_metadata_boundaries_are_fail_closed() -> None:
    """Scalar validators reject malformed values without reflecting input text."""
    with pytest.raises(ValueError, match="control characters"):
        AssessmentSpecError("valid_code", "$\n", "safe message")
    with pytest.raises(ValueError, match="at most"):
        AssessmentSpecError("valid_code", "$", "x" * 513)
    assert _error(lambda: validation._require_utf8("\ud800", "$.text")).code == (
        "invalid_utf8_text"
    )
    assert _error(lambda: validation.descriptive_identifier(1, "sample_name")).code == (
        "invalid_sample_name"
    )
    assert _error(lambda: validation.bounded_text(1, "sample_text")).code == (
        "invalid_sample_text"
    )
    assert _error(lambda: validation.semantic_version("v1", "sample_version")).code == (
        "invalid_sample_version"
    )
    assert validation.bounded_text("safe text", "sample_text") == "safe text"
    with pytest.raises(AssessmentSpecError, match="schema_version"):
        validation.assessment_schema_version("2.0")
    assert validation.fingerprint("a" * 64, "digest") == "a" * 64
    assert _error(lambda: validation.fingerprint("not-a-digest", "digest")).code == (
        "invalid_digest"
    )
    assert validation.assessment_schema_version("1.0") == "1.0"
    assert validation.enum_value(_SampleEnum.ENABLED, _SampleEnum, "mode") is _SampleEnum.ENABLED
    assert validation.enum_value("enabled", _SampleEnum, "mode") is _SampleEnum.ENABLED
    assert _error(lambda: validation.enum_value("disabled", _SampleEnum, "mode")).code == (
        "invalid_mode"
    )
    assert validation.strict_boolean(True, "enabled") is True
    assert _error(lambda: validation.strict_boolean(1, "enabled")).code == "invalid_enabled"


def test_positive_integer_and_collection_contracts_cover_conversion_failures() -> None:
    """Integer and iterable validators distinguish type, bound, and iterator failures."""
    assert validation.bounded_positive_integer(2, "count", 3) == 2
    for value in (False, object(), 0, 4):
        assert _error(lambda value=value: validation.bounded_positive_integer(value, "count", 3)).code == (
            "invalid_count"
        )
    assert validation.bounded_values((1, 2), "values", minimum=1, maximum=2) == (1, 2)
    for value in ("text", object()):
        assert _error(lambda value=value: validation.bounded_values(value, "values", minimum=1, maximum=2)).code == (
            "invalid_values"
        )

    class _FailingIterator:
        def __iter__(self):
            return self

        def __next__(self):
            raise ValueError("iterator failure")

    assert _error(
        lambda: validation.bounded_values(_FailingIterator(), "values", minimum=1, maximum=2)
    ).code == "invalid_values"
    assert _error(
        lambda: validation.bounded_values((1, 2, 3), "values", minimum=1, maximum=2)
    ).code == "invalid_values"
    assert _error(
        lambda: validation.bounded_values((), "values", minimum=1, maximum=2)
    ).code == "invalid_values"


def test_sorted_identity_helpers_and_metadata_mapping_fail_closed() -> None:
    """Sorted identity helpers and mapping materialization preserve safe paths."""
    assert validation.sorted_identifiers(("beta_value", "alpha_value"), "names", minimum=1) == (
        "alpha_value",
        "beta_value",
    )
    assert _error(
        lambda: validation.sorted_identifiers(("alpha_value", "alpha_value"), "names", minimum=1)
    ).code == "duplicate_names"
    assert validation.sorted_fingerprints(("b" * 64, "a" * 64), "digests", minimum=1) == (
        "a" * 64,
        "b" * 64,
    )
    assert _error(
        lambda: validation.sorted_fingerprints(("a" * 64, "a" * 64), "digests", minimum=1)
    ).code == "duplicate_digests"
    assert _error(lambda: validation.freeze_json_value(_BadItemsMapping(), "$.metadata", depth=0, node_count=[0])).code == (
        "invalid_metadata_mapping"
    )
    assert _error(lambda: validation.freeze_json_value(_MalformedItemsMapping(), "$.metadata", depth=0, node_count=[0])).code == (
        "invalid_metadata_mapping"
    )
    assert _error(lambda: validation.freeze_json_value(_PartiallyBrokenMapping(), "$.metadata", depth=0, node_count=[0])).code == (
        "invalid_metadata_mapping"
    )
    assert _error(lambda: validation.freeze_json_value(_DuplicateItemsMapping(), "$.metadata", depth=0, node_count=[0])).code == (
        "duplicate_metadata_key"
    )
    assert _error(
        lambda: validation.freeze_json_value(
            {f"metadata_key_{index}": index for index in range(validation.MAX_METADATA_COLLECTION_VALUES + 1)},
            "$.metadata",
            depth=0,
            node_count=[0],
        )
    ).code == "metadata_collection_too_large"
    assert _error(
        lambda: validation.freeze_json_value("x", "$.metadata", depth=validation.MAX_METADATA_DEPTH + 1, node_count=[0])
    ).code == "metadata_depth_exceeded"
    assert _error(
        lambda: validation.freeze_json_value(None, "$.metadata", depth=0, node_count=[validation.MAX_METADATA_NODES])
    ).code == "metadata_node_budget_exceeded"


def test_canonical_artifact_and_size_boundaries_are_deterministic() -> None:
    """Package-owned artifacts hash canonically and oversized payloads are rejected."""
    assert canonical_json(_CanonicalContract()) == '{"contract_value":"stable"}'
    assert len(artifact_digest({"contract_value": "stable"})) == 64
    with pytest.raises(NotImplementedError):
        validation.CanonicalContract()._content_dict()
    assert validation._canonical_payload(_CanonicalContract()) == {"contract_value": "stable"}
    assert len(validation.artifact_digest({"contract_value": "stable"})) == 64
    with pytest.raises(AssessmentSpecError, match="unsupported canonical artifact"):
        validation._canonical_payload(object())
    with pytest.raises(AssessmentSpecError, match="canonical JSON"):
        canonical_json(["x" * 8_192] * 33)
