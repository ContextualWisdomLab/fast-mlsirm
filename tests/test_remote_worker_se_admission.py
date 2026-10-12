"""Worker SE transport preserves the public wrapper's admission contract.

Core results below are recording fixtures, not numerical SE acceptance.
"""
from contextlib import redirect_stdout
import copy
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
from fast_mlsirm import bifactor_grm, fitstats, remote_worker
from fast_mlsirm.remote_exec import (
    ExecutionFloatPath, RemoteJobEnvelope, RemoteJobFamily, RemoteRunManifest,
    SEED_DERIVATION_RULE, payload_identity_sha256,
)


def _payload():
    return dict(a_general=[1.0] * 4, a_specific=[1.0] * 4,
                threshold=[[0.0]] * 4,
                responses=[[0, 1, 0, 1], [1, 0, 1, 0]],
                specific_map=[0, 0, 1, 1], n_cat=2, n_specific=2,
                q_general=121, q_specific=241, fd_step=1e-4)


def _direct(payload):
    return bifactor_grm.bifactor_oakes_se(
        payload['a_general'], payload['a_specific'], payload['threshold'],
        payload['responses'], payload['specific_map'], payload['n_cat'],
        payload['n_specific'], q_general=payload['q_general'],
        q_specific=payload['q_specific'], fd_step=payload['fd_step'])


@pytest.fixture
def recording_core(monkeypatch):
    calls = []

    def binding(*args):
        calls.append(args)
        return dict(labels=['fixture'], information=[1.0], vcov=None, se=None,
                    positive_definite=False, non_pd_reason='recording fixture')

    monkeypatch.setattr(fitstats, '_core_module',
                        lambda: SimpleNamespace(bifactor_oakes_se=binding))
    return calls


_MALFORMED = [
    ('responses', [[0.5, 1, 0, 1], [1, 0, 1, 0]]),
    ('specific_map', [0.5, 0, 1, 1]),
    ('n_cat', 2.5), ('n_specific', 2.5),
    ('q_general', 121.5), ('q_specific', 241.5),
]


@pytest.mark.parametrize('field,value', _MALFORMED, ids=[p[0] for p in _MALFORMED])
@pytest.mark.parametrize('entry', ['function', 'envelope', 'main'])
def test_worker_rejects_fractional_inputs_before_binding(field, value, entry,
                                                       recording_core, monkeypatch):
    payload = _payload()
    payload[field] = value
    before = copy.deepcopy(payload)
    with pytest.raises(ValueError):
        _direct(payload)
    assert recording_core == []
    manifest = RemoteRunManifest(
        schema_version='1.0', library_version=remote_worker._library_version(),
        source_sha256='a' * 64, seed_derivation_rule=SEED_DERIVATION_RULE,
        float_path=ExecutionFloatPath.F64,
        payload_sha256=payload_identity_sha256(payload),
        integration_nodes_sha256='c' * 64)
    envelope = RemoteJobEnvelope(
        run_id='se_admission_control', family=RemoteJobFamily.SE_DERIVATIVES,
        unit_index=0, base_seed=1, payload_ref=manifest.payload_sha256,
        manifest=manifest)
    try:
        if entry == 'function':
            with pytest.raises(ValueError):
                remote_worker.execute_se_derivatives(payload)
        elif entry == 'envelope':
            with pytest.raises(ValueError):
                remote_worker.execute_envelope(envelope, payload)
        else:
            request = {'envelope': envelope.to_dict(), 'payload': payload}
            monkeypatch.setattr(remote_worker.sys, 'stdin', io.StringIO(json.dumps(request)))
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = remote_worker.main()
            record = json.loads(output.getvalue())
            assert exit_code == 0
            assert record['delivery_state'] == 'failed'
            assert record['error_message']
            assert 'result' not in record
    finally:
        assert payload == before
        assert recording_core == [], 'malformed caller reached numerical binding'


@pytest.mark.parametrize('shape', ['integers', 'integral-floats', 'negative-missing'])
def test_worker_valid_input_matches_direct_binding_transport(shape, recording_core):
    payload = _payload()
    if shape == 'integral-floats':
        payload['specific_map'] = [float(x) for x in payload['specific_map']]
        payload['responses'] = [[float(x) for x in row] for row in payload['responses']]
    elif shape == 'negative-missing':
        payload['responses'][0][0] = -1
    before = copy.deepcopy(payload)
    _direct(payload)
    direct_args = recording_core.pop()
    result = remote_worker.execute_se_derivatives(payload)
    assert len(recording_core) == 1
    for direct, via_worker in zip(direct_args, recording_core[0]):
        if isinstance(direct, np.ndarray):
            np.testing.assert_array_equal(direct, via_worker)
        else:
            assert direct == via_worker
    assert result['positive_definite'] is False
    assert result['se_sha256'] is None
    assert result['non_pd_reason'] == 'recording fixture'
    assert payload == before
