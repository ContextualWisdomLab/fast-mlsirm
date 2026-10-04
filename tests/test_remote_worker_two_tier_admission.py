"""Two-tier worker transmission must retain the public wrapper admission.

Recording core outputs verify transport, not calibration or recovery.
"""
from contextlib import redirect_stdout
import copy
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
from fast_mlsirm import fitstats, remote_worker, two_tier_grm
from fast_mlsirm.remote_exec import (
    ExecutionFloatPath, RemoteJobEnvelope, RemoteJobFamily, RemoteRunManifest,
    SEED_DERIVATION_RULE, payload_identity_sha256,
)


def _payload():
    return dict(responses=[[0, 1, 0, 1], [1, 0, 1, 0]],
                primary_map=[[True, False], [True, False],
                             [False, True], [False, True]],
                specific_map=[0, 0, 1, 1], n_cat=2, n_primary=2, n_specific=2,
                q_primary=121, q_specific=241, max_iter=2, tol=1e-4, n_starts=1)


def _direct(payload, seed):
    return two_tier_grm.fit_two_tier_grm(
        payload['responses'], payload['primary_map'], payload['specific_map'],
        payload['n_cat'], payload['n_primary'], payload['n_specific'],
        payload['q_primary'], payload['q_specific'], payload['max_iter'],
        payload['tol'], payload['n_starts'], seed)


def _envelope(payload):
    manifest = RemoteRunManifest(
        schema_version='1.0', library_version=remote_worker._library_version(),
        source_sha256='a' * 64, seed_derivation_rule=SEED_DERIVATION_RULE,
        float_path=ExecutionFloatPath.F64,
        payload_sha256=payload_identity_sha256(payload),
        integration_nodes_sha256='c' * 64)
    return RemoteJobEnvelope(
        run_id='two_tier_admission_control', family=RemoteJobFamily.TWO_TIER,
        unit_index=0, base_seed=1, payload_ref=manifest.payload_sha256,
        manifest=manifest)


@pytest.fixture
def recording_core(monkeypatch):
    calls = []

    def binding(*args):
        calls.append(args)
        return dict(a_primary=[1., 0., 1., 0., 0., 1., 0., 1.],
                    a_specific=[1.] * 4, threshold=[0.] * 4,
                    phi=[1., 0., 0., 1.], theta_p_eap=[0.] * 4,
                    theta_p_sd=[1.] * 4, category_counts=[1] * 8,
                    loglik_trace=[-2.], n_iter=1, converged=False,
                    termination_reason='recording fixture', final_loglik_change=1.,
                    best_start=0, n_parameters=12, primary_identification='correlated')

    monkeypatch.setattr(fitstats, '_core_module',
                        lambda: SimpleNamespace(fit_two_tier_grm=binding))
    return calls


_BAD = [
    ('responses', [[0.5, 1, 0, 1], [1, 0, 1, 0]]),
    ('specific_map', [0.5, 0, 1, 1]),
    ('primary_map', [[0.5, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
    ('n_cat', 2.5), ('n_primary', 2.5), ('n_specific', 2.5),
    ('q_primary', 121.5), ('q_specific', 241.5), ('max_iter', 2.5),
    ('n_starts', 1.5), ('q_primary', True), ('tol', True),
]


@pytest.mark.parametrize('field,value', _BAD,
                         ids=['responses', 'specific_map', 'primary_map', 'n_cat',
                              'n_primary', 'n_specific', 'q_primary', 'q_specific',
                              'max_iter', 'n_starts', 'bool-q', 'bool-tol'])
@pytest.mark.parametrize('entry', ['function', 'envelope', 'main'])
def test_two_tier_worker_retains_direct_wrapper_rejection(
        field, value, entry, recording_core, monkeypatch):
    payload = _payload()
    payload[field] = value
    before = copy.deepcopy(payload)
    envelope = _envelope(payload)
    with pytest.raises(ValueError):
        _direct(payload, envelope.unit_seed())
    assert recording_core == []
    try:
        if entry == 'function':
            with pytest.raises(ValueError):
                remote_worker.execute_two_tier(payload, envelope.unit_seed())
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
        assert recording_core == [], 'malformed two-tier input reached calibration binding'


@pytest.mark.parametrize('shape', ['integers', 'integral-responses',
                                  'negative-missing', 'integral-controls', 'specific-free'])
def test_two_tier_worker_valid_input_matches_direct_transport(shape, recording_core):
    payload = _payload()
    if shape == 'integral-responses':
        payload['responses'] = [[float(x) for x in row] for row in payload['responses']]
    elif shape == 'negative-missing':
        payload['responses'][0][0] = -1
    elif shape == 'integral-controls':
        for field in ['n_cat', 'n_primary', 'n_specific', 'q_primary',
                      'q_specific', 'max_iter', 'n_starts']:
            payload[field] = float(payload[field])
    elif shape == 'specific-free':
        payload['specific_map'][0] = -1
    before = copy.deepcopy(payload)
    envelope = _envelope(payload)
    _direct(payload, envelope.unit_seed())
    direct_args = recording_core.pop()
    result = remote_worker.execute_envelope(envelope, payload)
    assert len(recording_core) == 1 and len(direct_args) == len(recording_core[0])
    for direct, worker in zip(direct_args, recording_core[0]):
        if isinstance(direct, np.ndarray):
            np.testing.assert_array_equal(direct, worker)
        else:
            assert direct == worker
    assert recording_core[0][9:11] == (121, 241)
    assert recording_core[0][14] == envelope.unit_seed()
    assert result['converged'] is False and result['final_loglik_change'] == 1.0
    assert payload == before
