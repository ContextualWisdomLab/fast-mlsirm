"""Public commit must validate before recording stream/hash side effects.

Recording Valkey only, not real-server or numerical acceptance.
"""
from dataclasses import replace
import copy
import hashlib
import json

import pytest
from fast_mlsirm import remote_exec as module
from test_remote_exec_valkey import FakeValkey, _completed_outcome


@pytest.mark.parametrize('occupied', [False, True])
@pytest.mark.parametrize('case', ['finite-tamper', 'forged-nan'])
def test_public_commit_rejects_before_any_write(occupied, case):
    valid = _completed_outcome(38, {'objective': 1.0})
    result = {'objective': 2.0 if case == 'finite-tamper' else float('nan')}
    digest = valid.output_identity_sha256
    if case == 'forged-nan':
        digest = hashlib.sha256(json.dumps(result, ensure_ascii=False, sort_keys=True,
                                           separators=(',', ':')).encode()).hexdigest()
    candidate = replace(valid, result=result, output_identity_sha256=digest)
    key = valid.envelope_fingerprint
    client = FakeValkey()
    store = module.ValkeyStreamsOutcomeStore(client, stream='outcomes',
                                            group='drivers', consumer='fixture', block_ms=0)
    if occupied:
        client.hashes['outcomes:committed'] = {key: json.dumps(valid.to_dict())}
    before = copy.deepcopy((client.streams, client.hashes, client.acked))
    try:
        with pytest.raises(ValueError, match='stored completed result'):
            store.commit_success(key, candidate)
    finally:
        assert (client.streams, client.hashes, client.acked) == before


@pytest.mark.parametrize('occupied', [False, True])
def test_public_valid_commit_appends_and_keeps_first_winner(occupied):
    first = _completed_outcome(39, {'objective': 1.0})
    later = _completed_outcome(39, {'objective': 2.0})
    key = first.envelope_fingerprint
    client = FakeValkey()
    store = module.ValkeyStreamsOutcomeStore(client, stream='outcomes',
                                            group='drivers', consumer='fixture', block_ms=0)
    if occupied:
        client.hashes['outcomes:committed'] = {key: json.dumps(first.to_dict())}
    winner = store.commit_success(key, later)
    assert winner == (first if occupied else later)
    records = client.streams['outcomes']
    assert len(records) == 1
    assert records[0][1]['fingerprint'] == key
    assert json.loads(records[0][1]['outcome']) == later.to_dict()
    assert module._valkey_outcome(client.hashes['outcomes:committed'][key], fingerprint=key) == winner
    assert client.acked == []


import copy
from dataclasses import replace

import pytest
from fast_mlsirm import remote_exec as module
from test_remote_exec_valkey import FakeValkey, _completed_outcome


@pytest.mark.parametrize('occupied', [False, True])
@pytest.mark.parametrize('access', ['public', 'persist'])
@pytest.mark.parametrize('shape', ['ascii-size', 'utf8-size', 'depth'])
def test_bounded_writer_rejects_before_stream_or_hash_mutation(occupied, access, shape):
    limit = module._MAX_VALKEY_OUTCOME_JSON_BYTES
    if shape == 'depth':
        nested = 1.0
        for _ in range(module._MAX_VALKEY_OUTCOME_JSON_DEPTH - 1):
            nested = [nested]
        result = {'nested': nested}
        error = 'Valkey outcome JSON nests deeper'
    else:
        result = {'pad': 'x' * limit if shape == 'ascii-size' else '한' * (limit // 3 + 1)}
        error = 'Valkey outcome JSON exceeds'
    candidate = _completed_outcome(43, result)
    raw = module.ValkeyStreamsOutcomeStore._serialize_outcome(candidate)
    if shape == 'depth':
        assert module._json_nesting_exceeds(raw, module._MAX_VALKEY_OUTCOME_JSON_DEPTH)
        assert len(raw.encode('utf-8')) <= limit
    else:
        assert len(raw.encode('utf-8')) > limit
        if shape == 'utf8-size':
            assert len(raw) < limit
    assert candidate.output_identity_sha256 == module.result_identity_sha256(result)
    first = _completed_outcome(43, {'first': True})
    key = candidate.envelope_fingerprint
    client = FakeValkey()
    store = module.ValkeyStreamsOutcomeStore(client, stream='outcomes',
                                            group='drivers', consumer='fixture', block_ms=0)
    if occupied:
        client.hashes['outcomes:committed'] = {key: store._serialize_outcome(first)}
    before = copy.deepcopy((client.streams, client.hashes, client.acked))
    try:
        with pytest.raises(ValueError, match=error):
            if access == 'public':
                store.commit_success(key, candidate)
            else:
                store._persist_committed(key, candidate)
    finally:
        assert (client.streams, client.hashes, client.acked) == before


@pytest.mark.parametrize('occupied', [False, True])
@pytest.mark.parametrize('access', ['public', 'persist'])
@pytest.mark.parametrize('boundary', ['exact-byte-limit', 'exact-depth-limit'])
def test_bounded_writer_boundary_control_preserves_first_success(occupied, access, boundary):
    empty = _completed_outcome(44, {'pad': ''})
    if boundary == 'exact-byte-limit':
        overhead = len(module.ValkeyStreamsOutcomeStore._serialize_outcome(empty).encode('utf-8'))
        result = {'pad': 'x' * (module._MAX_VALKEY_OUTCOME_JSON_BYTES - overhead)}
    else:
        nested = 1.0
        for _ in range(module._MAX_VALKEY_OUTCOME_JSON_DEPTH - 2):
            nested = [nested]
        result = {'nested': nested}
    candidate = replace(empty, result=result,
                        output_identity_sha256=module.result_identity_sha256(result))
    raw = module.ValkeyStreamsOutcomeStore._serialize_outcome(candidate)
    if boundary == 'exact-byte-limit':
        assert len(raw.encode('utf-8')) == module._MAX_VALKEY_OUTCOME_JSON_BYTES
    else:
        assert not module._json_nesting_exceeds(raw, module._MAX_VALKEY_OUTCOME_JSON_DEPTH)
        assert module._json_nesting_exceeds(raw, module._MAX_VALKEY_OUTCOME_JSON_DEPTH - 1)
    assert module._valkey_successful_outcome(raw, fingerprint=candidate.envelope_fingerprint) == candidate
    first = _completed_outcome(44, {'first': True})
    key = candidate.envelope_fingerprint
    client = FakeValkey()
    store = module.ValkeyStreamsOutcomeStore(client, stream='outcomes',
                                            group='drivers', consumer='fixture', block_ms=0)
    if occupied:
        client.hashes['outcomes:committed'] = {key: store._serialize_outcome(first)}
    winner = store.commit_success(key, candidate) if access == 'public' else store._persist_committed(key, candidate)
    assert winner == (first if occupied else candidate)
    assert module._valkey_successful_outcome(client.hashes['outcomes:committed'][key], fingerprint=key) == winner
    assert client.acked == []
    if access == 'public':
        assert len(client.streams['outcomes']) == 1
        assert client.streams['outcomes'][0][1] == {'fingerprint': key, 'outcome': raw}
    else:
        assert client.streams == {}


def test_module_scope_includes_adapter_and_shared_batch_admission():
    assert 'Valkey/Redis transport is intentionally out of scope' not in module.__doc__
    assert 'ValkeyStreamsOutcomeStore' in module.__doc__
    assert 'ValkeyStreamsBackend.run_batch' in module.__doc__
    assert '_admit_payload_batch' in module.__doc__
    assert 'does not certify' in module.__doc__
