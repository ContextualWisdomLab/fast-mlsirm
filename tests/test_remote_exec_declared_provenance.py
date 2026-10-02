# Copyright (c) 2026 ContextualWisdomLab. All rights reserved.
# SPDX-License-Identifier: MIT
"""Declared identity must never be presented as measured attestation."""
from dataclasses import replace
import io
import json
import socket

import pytest

from test_remote_exec import _manifest, _payload_manifest, _envelope
from fast_mlsirm import remote_exec, remote_worker
from fast_mlsirm.remote_exec import RemoteWorkerProvenance, local_worker_provenance


def test_declared_identity_is_explicit_in_json_and_legacy_restore():
    """Legacy records restore as unverified declarations, not attestations."""
    provenance = local_worker_provenance(
        _manifest(), requested_device='cpu', effective_device='cpu',
        wall_clock_seconds=0.0,
    )
    encoded = provenance.to_dict()
    assert encoded['identity_verification'] == 'declared_unverified'
    legacy = {key: value for key, value in encoded.items() if key != 'identity_verification'}
    restored = RemoteWorkerProvenance(**legacy)
    assert restored.to_dict()['identity_verification'] == 'declared_unverified'


def test_caller_cannot_promote_declaration_to_attested():
    """No measured producer contract exists to support a stronger label."""
    provenance = local_worker_provenance(
        _manifest(), requested_device='cpu', effective_device='cpu',
        wall_clock_seconds=0.0,
    )
    with pytest.raises(ValueError, match='declared_unverified'):
        replace(provenance, identity_verification='attested')


@pytest.mark.parametrize('requested,effective', [('gpu', 'cpu'), ('cpu', 'gpu'), ('auto', 'cpu')])
def test_subprocess_rejects_unsupported_device_declarations_before_dispatch(monkeypatch, requested, effective):
    """A device label must not claim unsupported execution-path selection."""
    payload: dict[str, object] = {'placeholder': True}
    envelope = _envelope(manifest=_payload_manifest(payload))
    def forbidden_spawn(*args, **kwargs):
        raise AssertionError('unsupported declaration reached worker spawn')
    monkeypatch.setattr(remote_exec, '_invoke_worker_process', forbidden_spawn)
    with pytest.raises(ValueError, match='only cpu device declarations'):
        remote_exec.SubprocessExecutor(socket.gethostname()).run_batch(
            (envelope,), worker_manifest=envelope.manifest, payload=payload,
            requested_device=requested, effective_device=effective,
        )


@pytest.mark.parametrize('requested,effective', [('gpu', 'cpu'), ('cpu', 'gpu'), ('auto', 'cpu')])
def test_worker_rejects_unsupported_device_declarations_before_execution(monkeypatch, capsys, requested, effective):
    """Direct CLI requests cannot bypass the driver device-claim gate."""
    payload: dict[str, object] = {'placeholder': True}
    envelope = _envelope(manifest=_payload_manifest(payload))
    request = {'envelope': envelope.to_dict(), 'payload': payload,
               'requested_device': requested, 'effective_device': effective}
    executed = []
    def record_execution(*args):
        executed.append(True)
        return {'control': True}
    monkeypatch.setattr(remote_worker.sys, 'stdin', io.StringIO(json.dumps(request)))
    monkeypatch.setattr(remote_worker, 'execute_envelope', record_execution)
    assert remote_worker.main([]) == 0
    reply = json.loads(capsys.readouterr().out)
    assert reply['delivery_state'] == 'failed'
    assert 'only cpu device declarations' in reply['error_message']
    assert executed == []


def test_worker_cpu_declaration_preserves_legacy_request_control(monkeypatch, capsys):
    """Omitted labels mean the same unverified CPU declaration as before."""
    payload: dict[str, object] = {'placeholder': True}
    envelope = _envelope(manifest=_payload_manifest(payload))
    monkeypatch.setattr(remote_worker.sys, 'stdin', io.StringIO(json.dumps(
        {'envelope': envelope.to_dict(), 'payload': payload})))
    monkeypatch.setattr(remote_worker, 'execute_envelope', lambda *args: {'control': True})
    assert remote_worker.main([]) == 0
    assert json.loads(capsys.readouterr().out)['delivery_state'] == 'completed'
