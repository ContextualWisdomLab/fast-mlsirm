"""Sibling wrapper transport: synthetic recording results, no fit or SE computation."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from test_two_tier_expected_raw_layout import seam


@pytest.mark.parametrize('route', ['fit', 'oakes'])
@pytest.mark.parametrize('control', ['q_primary', 'q_specific'])
@pytest.mark.parametrize('q', [121, 241, 2**53 + 1, np.uint64(2**53 + 1)],
                         ids=['121', '241', 'large-python', 'large-numpy'])
def test_sibling_node_transport(seam, monkeypatch, route, control, q):
    """Exercise each real sibling wrapper without running its numerical engine."""
    module, unused = seam
    recorded = []
    y = np.array([[0, 1], [1, 0]])
    pmap = np.ones((2, 1), dtype=bool)
    smap = np.zeros(2, dtype=np.int64)
    ag = np.array([[1.2], [0.8]])
    asp = np.array([0.5, 0.4])
    threshold = np.array([[0.2], [-0.2]])
    phi = np.eye(1)
    caller = [y, pmap, smap, ag, asp, threshold, phi]
    saved = [a.copy() for a in caller]

    def fit_binding(*args):
        recorded.append(args)
        return dict(a_primary=ag.copy(), a_specific=asp.copy(),
                    threshold=threshold.copy(), phi=phi.copy(),
                    theta_p_eap=np.zeros((2, 1)), theta_p_sd=np.ones((2, 1)),
                    category_counts=np.ones((2, 2), dtype=int), loglik_trace=[-3.],
                    n_iter=1, converged=False, termination_reason='max_iter_reached',
                    final_loglik_change=1., best_start=0, n_parameters=6,
                    primary_identification='correlated')

    def oakes_binding(*args):
        recorded.append(args)
        return dict(labels=['synthetic'], information=[2.], vcov=[0.5],
                    se=[np.sqrt(0.5)], positive_definite=True, non_pd_reason=None)

    core = SimpleNamespace(fit_two_tier_grm=fit_binding, two_tier_oakes_se=oakes_binding)
    monkeypatch.setattr(sys.modules[module.__package__+'.fitstats'], '_core_module', lambda: core)

    def call(value):
        qs = dict(q_primary=121, q_specific=241)
        qs[control] = value
        if route == 'fit':
            return module.fit_two_tier_grm(y, pmap, smap, 2, 1, 1,
                max_iter=1, tol=1e-6, n_starts=1, seed=17, **qs)
        return module.two_tier_oakes_se(ag, asp, threshold, phi, y, pmap, smap,
            2, 1, 1, fd_step=1e-4, **qs)

    call(121)
    baseline = recorded[-1]
    call(q)
    changed = recorded[-1]
    assert len(recorded) == 2 and unused == []
    position = {'fit': {'q_primary': 9, 'q_specific': 10},
                'oakes': {'q_primary': 13, 'q_specific': 14}}[route][control]
    assert type(changed[position]) is int
    assert changed[position] == int(q)
    for i, (before, after) in enumerate(zip(baseline, changed)):
        if i == position:
            continue
        if isinstance(before, np.ndarray):
            np.testing.assert_array_equal(before, after)
        else:
            assert before == after
    for actual, original in zip(caller, saved):
        np.testing.assert_array_equal(actual, original)
