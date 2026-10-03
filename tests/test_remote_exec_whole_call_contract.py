# Copyright (c) 2026 ContextualWisdomLab. All rights reserved.
# SPDX-License-Identifier: MIT
"""Whole-call identifiers are not numerical internal-shard identifiers."""
from pathlib import Path

import pytest

from test_remote_exec import _envelope, _payload_manifest
from fast_mlsirm.remote_exec import (
    INTERNALLY_UNSHARDABLE_REMOTE_JOB_FAMILIES,
    LoopbackExecutor,
    RemoteJobFamily,
)
from fast_mlsirm import remote_worker


def test_document_describes_nonzero_whole_call_identifier_and_batch_limit():
    """Documentation must not promise a zero-index or global batch guard."""
    text = (Path(__file__).parents[1] / 'docs/orchestration/remote-family-io-equivalence-2072.md').read_text()
    assert '`unit_index` must be 0 per run' not in text
    assert 'nonnegative whole-call identifier' in text
    assert 'does not enforce a run-wide invariant across separate batches' in text


@pytest.mark.parametrize('family', sorted(INTERNALLY_UNSHARDABLE_REMOTE_JOB_FAMILIES))
def test_separate_batches_preserve_nonzero_whole_call_identifiers(family):
    """A batch guard cannot infer internal partitioning from an index alone."""
    manifest = _payload_manifest({'placeholder': True})
    executor = LoopbackExecutor()
    handled = []
    for index in (3, 7):
        envelope = _envelope(family=RemoteJobFamily(family), unit_index=index, manifest=manifest)
        outcomes = executor.run_batch(
            (envelope,), lambda record, seed: handled.append(record.unit_index),
            worker_manifest=manifest,
        )
        assert outcomes[0].unit_index == index
    assert handled == [3, 7]


def test_direct_worker_keeps_nonzero_identifier_for_one_complete_call(monkeypatch):
    """The direct worker adapter forwards one whole call, not a shard."""
    payload: dict[str, object] = {'placeholder': True}
    envelope = _envelope(family=RemoteJobFamily.EM_M_STEP, unit_index=3,
                         manifest=_payload_manifest(payload))
    seen = []
    def whole_call(actual_payload, seed):
        seen.append((actual_payload, seed))
        return {'whole_call': True}
    monkeypatch.setattr(remote_worker, 'execute_em_m_step', whole_call)
    assert remote_worker.execute_envelope(envelope, payload) == {'whole_call': True}
    assert seen == [(payload, envelope.unit_seed())]
