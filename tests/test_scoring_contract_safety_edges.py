"""Adversarial callback and serialization coverage for contract safety wrappers."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum

import pytest

from fast_mlsirm.scoring import AssessmentSpecError
import fast_mlsirm.scoring._contract_safety as safety
import fast_mlsirm.scoring._validation as base


class _ErrorEnum(Enum):
    """Enum whose missing-value hook exposes the package error branch."""

    VALUE = "value"

    @classmethod
    def _missing_(cls, value):
        raise base.assessment_error("invalid_mode", "$.mode", "unsupported mode")


class _ErrorIterable:
    """Iterable whose iterator construction raises a domain error."""

    def __iter__(self):
        raise base.assessment_error("invalid_values", "$.values", "bad iterator")


class _ExplodingIterable:
    """Iterable that fails after yielding one value."""

    def __iter__(self):
        yield 1
        raise RuntimeError("iteration failed")


class _BadItemsMapping(Mapping):
    """Mapping whose item iterator cannot be inspected."""

    def __getitem__(self, key):
        return 1

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0

    def items(self):
        raise RuntimeError("items failed")


class _MalformedItemsMapping(Mapping):
    """Mapping whose item stream contains a non-pair entry."""

    def __getitem__(self, key):
        return 1

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0

    def items(self):
        return iter((("key_only",),))


class _DuplicateItemsMapping(Mapping):
    """Mapping-like value with duplicate keys in its item stream."""

    def __getitem__(self, key):
        return 1

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0

    def items(self):
        return iter((("same_key", 1), ("same_key", 2)))


class _BadLengthList(list):
    """List whose length callback fails at the metadata trust boundary."""

    def __len__(self):
        raise RuntimeError("length failed")


class _BadIterationList(list):
    """List whose iterator callback fails after a valid length check."""

    def __iter__(self):
        raise RuntimeError("iteration failed")


class _ContentContract(base.CanonicalContract):
    """Minimal canonical contract for complete-payload serialization branches."""

    def __init__(self, content):
        self.content = content

    def _content_dict(self):
        return {"content": self.content}


def _error(action):
    """Return the structured error raised by one safety action."""
    with pytest.raises(AssessmentSpecError) as captured:
        action()
    return captured.value


def test_wrappers_preserve_domain_errors_and_hide_callback_failures(monkeypatch):
    """Wrapper validators keep stable errors for both domain and callback failures."""
    assert _error(lambda: safety.bounded_text("", "sample_text")).code == "invalid_sample_text"
    assert _error(lambda: safety.semantic_version("v1", "sample_version")).code == "invalid_sample_version"
    monkeypatch.setattr(
        safety,
        "_text",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            base.assessment_error("invalid_sample_text", "$.sample_text", "invalid")
        ),
    )
    assert _error(lambda: safety.bounded_text("safe", "sample_text")).code == "invalid_sample_text"
    monkeypatch.setattr(
        safety,
        "_semantic_version",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            base.assessment_error("invalid_sample_version", "$.sample_version", "invalid")
        ),
    )
    assert _error(lambda: safety.semantic_version("1.0.0", "sample_version")).code == "invalid_sample_version"
    assert _error(lambda: safety.enum_value("other", _ErrorEnum, "mode")).code == "invalid_mode"
    assert _error(lambda: safety.bounded_values(_ErrorIterable(), "values", minimum=1, maximum=2)).code == "invalid_values"
    assert _error(lambda: safety.bounded_values(_ExplodingIterable(), "values", minimum=1, maximum=2)).code == "invalid_values"

    def _raise_fingerprint(*args, **kwargs):
        raise RuntimeError("fingerprint callback failed")

    monkeypatch.setattr(base, "fingerprint", _raise_fingerprint)
    assert _error(lambda: safety.sorted_fingerprints(("a" * 64,), "digests", minimum=1)).code == "invalid_digests"


def test_metadata_preflight_rejects_mapping_and_collection_callback_failures():
    """Metadata copying fails closed for malformed, duplicate, and broken containers."""
    for value, code in (
        (_BadItemsMapping(), "invalid_metadata_mapping"),
        (_MalformedItemsMapping(), "invalid_metadata_mapping"),
        (_DuplicateItemsMapping(), "duplicate_metadata_key"),
        (_BadLengthList([1]), "invalid_metadata_collection"),
        (_BadIterationList([1]), "invalid_metadata_collection"),
    ):
        assert _error(lambda value=value: safety.freeze_metadata({"nested": value})).code == code


def test_complete_contract_serialization_rejects_invalid_utf8_and_oversize():
    """The contract-specific serializer enforces UTF-8 and complete-size limits."""
    assert safety.canonical_json(_ContentContract("stable")) == '{"content":"stable"}'
    assert _error(lambda: safety.canonical_json(_ContentContract("\ud800"))).code == "invalid_utf8_text"
    oversized = "x" * (base.MAX_CANONICAL_JSON_CHARACTERS + 1)
    assert _error(lambda: safety.canonical_json(_ContentContract(oversized))).code == "canonical_json_too_large"
