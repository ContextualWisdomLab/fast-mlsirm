"""Checksum-valid synthetic state regressions; not execution attestations."""
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest
from fast_mlsirm import _core


def seed():
    # Tests use the same fixture on both frozen and repaired wheels. Source binding
    # is explicit; these states are documentary, not claims of fitted estimates.
    root = Path(os.environ.get('FMLS_STATE_SOURCE_ROOT', str(Path(__file__).resolve().parents[1]))) / 'crates/mlsirm-core/src'
    source = hashlib.sha256(b''.join((root / p).read_bytes() for p in
        ['mask_grm.rs', 'bifactor_grm.rs', 'mask_grm_state.rs'])).hexdigest()
    s: dict[str, Any] = dict(y=[0,1,1,0,0,1,1,0], observed=None, groups=[0,0,1,1], n_groups=2,
        n_items=2, n_cat=2, anchor=[True,True], specific_map=[-1,-1], n_specific=0,
        embedded_start=None, embedded_parent=None,
        cfg=dict(q_general=7,q_specific=1,max_iter=1,tol=1e-8,n_starts=1,seed=42,
            newton_iter=2,ridge=1e-6,slope_prior='None',estimate_specific_vars=False,
            estimate_specific_means=False,device='Cpu'),
        result=dict(scoring_posterior_gpu=False,start_outcomes=[],dispatch=[],
            dispatch_complete=False,m_step_gpu_dispatches=0,a_general=[[1.,1.],[1.,1.]],
            a_specific=[[0.,0.],[0.,0.]],threshold=[[0.,0.],[0.,0.]],
            general_mean=[0.,0.],general_sd=[1.,1.],specific_sd=[[],[]],specific_mean=[[],[]],
            theta_g_eap=[0.]*4,theta_g_sd=[1.]*4,group_category_counts=[[1]*4,[1]*4],
            loglik_trace=[-10.,-9.],n_iter=1,converged=False,termination_reason='max_iter_reached',
            final_loglik_change=1.,best_start=0,n_parameters=6,slope_prior='None',
            em_objective_trace=[-10.,-9.]))
    return source, s


def envelope(source, s):
    text = json.dumps(s, separators=(',', ':'), allow_nan=False)
    return json.dumps(dict(schema=1,source=source,sha256=hashlib.sha256(text.encode()).hexdigest(),state=text))


def specific(s):
    s['specific_map'] = [0,0]
    s['n_specific'] = 1
    s['cfg']['q_specific'] = 3
    s['result'].update(a_specific=[[.5,.7],[.5,.7]], specific_mean=[[0.],[0.]],
                       specific_sd=[[1.],[1.]], n_parameters=8)


@pytest.mark.parametrize('bifactor', [False, True])
@pytest.mark.parametrize('cell', [0, -1])
@pytest.mark.parametrize('api', ['sequential_dif_restore_summary','sequential_dif_roundtrip_private'])
def test_observed_category_rejected(bifactor, cell, api):
    source, s = seed()
    if bifactor:
        specific(s)
    s['y'][cell] = s['n_cat']
    with pytest.raises(ValueError, match='observed response categories must be < n_cat'):
        getattr(_core, api)(envelope(source,s))


MUTATIONS = [
    ('reference_mean', ('general_mean',0), 123.),
    ('reference_sd', ('general_sd',0), -2.),
    ('focal_sd_zero', ('general_sd',1), 0.),
    ('anchor_general', ('a_general',1,0), 456.),
    ('anchor_specific', ('a_specific',1,0), 456.),
    ('anchor_threshold', ('threshold',1,0), 456.),
    ('reference_specific_mean', ('specific_mean',0,0), 1.),
    ('reference_specific_sd', ('specific_sd',0,0), 2.),
    ('fixed_specific_mean', ('specific_mean',1,0), 1.),
    ('fixed_specific_sd', ('specific_sd',1,0), 2.),
    ('posterior_sd_negative', ('theta_g_sd',0), -1.),
]


@pytest.mark.parametrize('name,path,value', MUTATIONS, ids=[x[0] for x in MUTATIONS])
@pytest.mark.parametrize('api', ['sequential_dif_restore_summary','sequential_dif_roundtrip_private'])
def test_fixed_coordinates_and_scores_rejected(name, path, value, api):
    source, s = seed()
    specific(s)
    dst = s['result']
    for key in path[:-1]:
        dst = dst[key]
    dst[path[-1]] = value
    with pytest.raises(ValueError):
        getattr(_core, api)(envelope(source,s))


@pytest.mark.parametrize('case', ['counts_width','counts_values','trace_empty','trace_length','objective_length','inactive_specific'])
def test_saved_structure_rejected(case):
    source, s = seed()
    r = s['result']
    if case == 'counts_width': r['group_category_counts'][0].pop()
    if case == 'counts_values': r['group_category_counts'][0][0] += 1
    if case == 'trace_empty': r['loglik_trace'] = []
    if case == 'trace_length': r['n_iter'] += 1
    if case == 'objective_length': r['em_objective_trace'].pop()
    if case == 'inactive_specific': r['a_specific'][0][0] = .5
    with pytest.raises(ValueError):
        _core.sequential_dif_roundtrip_private(envelope(source,s))


@pytest.mark.parametrize('bifactor', [False, True])
@pytest.mark.parametrize('missing', [False, True])
def test_valid_documentary_state_preserved(bifactor, missing):
    source, s = seed()
    if bifactor:
        specific(s)
        s['cfg'].update(estimate_specific_means=True,estimate_specific_vars=True)
        s['result']['specific_mean'][1][0] = .2
        s['result']['specific_sd'][1][0] = 1.2
        s['result']['n_parameters'] += 2
    # Zero posterior SD is legitimate for a degenerate posterior.
    s['result']['theta_g_sd'][0] = 0.
    if missing:
        s['observed'] = [False]+[True]*7
        s['y'][0] = 999
        s['result']['group_category_counts'][0][0] -= 1
    raw = envelope(source,s)
    exported = _core.sequential_dif_roundtrip_private(raw)
    assert json.loads(json.loads(exported)['state']) == s
    summary = json.loads(_core.sequential_dif_restore_summary(raw))
    assert summary['general_mean'] == s['result']['general_mean']
    # Persistence is trusted metadata, not GPU execution attestation.
    assert summary['converged'] is False


def test_free_item_coordinates_preserved():
    source,s=seed()
    s['anchor'][1]=False
    s['result']['a_general'][1][1]=1.5
    s['result']['threshold'][1][1]=.2
    s['result']['n_parameters']+=2
    out=json.loads(_core.sequential_dif_roundtrip_private(envelope(source,s)))
    assert json.loads(out['state'])==s


@pytest.mark.parametrize('field', ['general_mean','general_sd','theta_g_eap','theta_g_sd'])
@pytest.mark.parametrize('literal', ['1e400','-1e400'])
def test_nonfinite_json_number_rejected(field, literal):
    source, s = seed()
    s['result'][field][0] = 'NONFINITE_NUMBER'
    # Valid JSON numeric syntax, overflowing binary64. Recompute the checksum
    # so parser/value validation, rather than envelope integrity, must refuse.
    text = json.dumps(s, separators=(',', ':')).replace('"NONFINITE_NUMBER"', literal)
    raw = json.dumps(dict(schema=1, source=source,
                         sha256=hashlib.sha256(text.encode()).hexdigest(), state=text))
    with pytest.raises(ValueError):
        _core.sequential_dif_roundtrip_private(raw)


def test_positive_reference_sd_drift_rejected():
    source, s = seed()
    s['result']['general_sd'][0] = 2.
    with pytest.raises(ValueError, match='latent moments'):
        _core.sequential_dif_roundtrip_private(envelope(source,s))


def test_reference_constraints_preserve_negative_zero():
    source, s = seed()
    specific(s)
    s['result']['general_mean'][0] = -0.0
    s['result']['specific_mean'][0][0] = -0.0
    raw = _core.sequential_dif_roundtrip_private(envelope(source,s))
    assert _core.sequential_dif_roundtrip_private(raw) == raw


@pytest.mark.parametrize('free_variance', [False, True])
def test_native_bifactor_producer_roundtrip(free_variance):
    # Genuine small synthetic native fit, not a scientific convergence claim.
    request = dict(y=[(p + i) % 4 for p in range(32) for i in range(4)],
        group_id=[p % 2 for p in range(32)], n_groups=2, n_cat=4,
        item_ids=['a','b','c','d'], anchor=[True]*4, q_general=5,
        max_iter=1,tol=1e-8,n_starts=1,seed=42,newton_iter=2,ridge=1e-6,
        device='cpu',alpha=.05,max_rounds=1,
        bifactor=dict(specific_map=[0,0,0,-1],n_specific=1,q_specific=3,
                      estimate_specific_means=False,estimate_specific_vars=free_variance),
        controls=dict(fd_step=.001,rank_atol=1e-9,rank_rtol=1e-7,
            objective_call_budget=10000,workspace_bytes=10000000,
            ll_roundoff_tolerance=1e-7,fit_ll_tolerance=1e-4,
            lr_backend_tolerance=1e-4,hessian_rtol=.01,require_gpu=False))
    raw = _core.sequential_dif_fit_private(json.dumps(request))
    assert _core.sequential_dif_roundtrip_private(raw) == raw
    restored = json.loads(_core.sequential_dif_restore_summary(raw))
    assert restored['specific_mean'] == [[0.],[0.]]
    assert restored['specific_sd'][0] == [1.]
    if not free_variance:
        assert restored['specific_sd'][1] == [1.]
