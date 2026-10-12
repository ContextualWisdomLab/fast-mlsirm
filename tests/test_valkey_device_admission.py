"""Proposed CPU-only declaration contract; recording client, not device execution."""
import copy

import pytest

from fast_mlsirm import remote_exec as module
from test_remote_exec_valkey import FakeValkey, _backend, _envelope, _manifest, _MC_PAYLOAD


@pytest.mark.parametrize('requested,effective', [
    ('gpu', 'cpu'), ('auto', 'cpu'), ('cpu', 'gpu'), ('cpu', 'auto'),
    ('CPU', 'cpu'), ('cpu', 'CPU'), (' cpu', 'cpu'), ('cpu', 'cpu '),
    (True, 'cpu'), ('cpu', None), ([], 'cpu'),
])
def test_valkey_device_declaration_rejected_before_payload_or_publication(monkeypatch, requested, effective):
    client = FakeValkey()
    backend = _backend(client)
    before = copy.deepcopy((client.streams, client.hashes, client.acked))
    admission_calls = []
    original = module._admit_payload_batch

    def recording_admission(*args, **kwargs):
        admission_calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, '_admit_payload_batch', recording_admission)
    try:
        with pytest.raises(ValueError):
            backend.run_batch((_envelope(),), worker_manifest=_manifest(), payload=_MC_PAYLOAD,
                              requested_device=requested, effective_device=effective)
    finally:
        assert (client.streams, client.hashes, client.acked) == before
        assert admission_calls == []
