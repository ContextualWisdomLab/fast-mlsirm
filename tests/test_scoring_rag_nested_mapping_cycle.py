"""Cycle contract for the RAG nested mapping snapshot."""

from __future__ import annotations

import pytest

from fast_mlsirm.scoring import AssessmentSpecError
from fast_mlsirm.scoring._rag_metadata_validation import _snapshot_rag_value


def test_snapshot_rejects_cyclic_nested_mapping() -> None:
    """A self-referential mapping fails before recursive materialization."""
    cyclic: dict[str, object] = {}
    cyclic["nested_mapping"] = cyclic

    with pytest.raises(AssessmentSpecError) as caught:
        _snapshot_rag_value(cyclic, "$.metadata")

    assert caught.value.code == "cyclic_metadata_reference"
    assert caught.value.path == "$.metadata.values[0]"
