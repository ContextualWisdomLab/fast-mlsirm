"""FIPC worker preserves the public wrapper's input admission.

Core returns are recording fixtures, not calibration or recovery evidence.
"""
from contextlib import redirect_stdout
import copy
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
from fast_mlsirm import polytomous, remote_worker
from fast_mlsirm.remote_exec import (
    ExecutionFloatPath, RemoteJobEnvelope, RemoteJobFamily, RemoteRunManifest,
    SEED_DERIVATION_RULE, payload_identity_sha256,
)


def _payload():
    return dict(responses=[[0, 1], [1, 0]], n_cat=2,
                anchor=[True, False], anchor_slope=[1.0, -1.0],
                anchor_cat_params=[[0.0], [0.0]], q_theta=121,
                max_iter=2, tol=1e-4)


def _direct(payload):
    return polytomous.fit_poly_fipc(
        payload['responses'], payload['n_cat'], payload['anchor'],
        payload['anchor_slope'], payload['anchor_cat_params'],
        q_theta=payload['q_theta'], max_iter=payload['max_iter'], tol=payload['tol'])


@pytest.fixture
def recording_core(monkeypatch):
    calls = []

    def binding(*args):
        calls.append(args)
        return dict(slope=[1.0, -1.0], cat_params=[[0.0], [0.0]],
                    mu=0.0, sigma=1.0, loglik=-2.0, n_iter=1,
                    converged=False, termination_reason='recording fixture',
                    loglik_trace=[-2.0], final_delta=1.0,
                    stopping_tolerance=1e-4)

    monkeypatch.setattr(polytomous, '_core_module',
                        lambda: SimpleNamespace(fit_poly_fipc=binding))
    return calls


_BAD = [
    ('responses', [[0.5, 1], [1, 0]]),
    ('n_cat', 2.5), ('q_theta', 121.5), ('max_iter', 2.5),
    ('q_theta', True), ('tol', True), ('tol', '0.0001'),
]


@pytest.mark.parametrize('field,value', _BAD,
                         ids=['responses', 'n_cat', 'q_theta', 'max_iter',
                              'bool-q', 'bool-tol', 'str-tol'])
@pytest.mark.parametrize('entry', ['function', 'envelope', 'main'])
def test_fipc_worker_retains_direct_wrapper_rejection(field, value, entry,
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
        run_id='fipc_admission_control', family=RemoteJobFamily.FIPC,
        unit_index=0, base_seed=1, payload_ref=manifest.payload_sha256,
        manifest=manifest)
    try:
        if entry == 'function':
            with pytest.raises(ValueError):
                remote_worker.execute_fipc(payload)
        elif entry == 'envelope':
            with pytest.raises(ValueError):
                remote_worker.execute_envelope(envelope, payload)
        else:
            monkeypatch.setattr(remote_worker.sys, 'stdin', io.StringIO(json.dumps(
                {'envelope': envelope.to_dict(), 'payload': payload})))
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
        assert recording_core == [], 'malformed FIPC input reached calibration binding'


@pytest.mark.parametrize('shape', ['integers', 'integral-floats', 'negative-missing'])
def test_fipc_worker_valid_input_matches_direct_transport(shape, recording_core):
    payload = _payload()
    if shape == 'integral-floats':
        payload['responses'] = [[float(x) for x in row] for row in payload['responses']]
    elif shape == 'negative-missing':
        payload['responses'][0][0] = -1
    before = copy.deepcopy(payload)
    _direct(payload)
    direct_args = recording_core.pop()
    result = remote_worker.execute_fipc(payload)
    assert len(recording_core) == 1
    for direct, worker in zip(direct_args, recording_core[0]):
        if isinstance(direct, np.ndarray):
            np.testing.assert_array_equal(direct, worker)
        else:
            assert direct == worker
    assert recording_core[0][8] == payload['q_theta'] == 121
    assert result['converged'] is False
    assert result['mu'] == 0.0 and result['sigma'] == 1.0
    assert payload == before
