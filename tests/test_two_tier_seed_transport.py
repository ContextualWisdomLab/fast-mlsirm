"""Seed transport contract only; synthetic fit result, no numerical fitting."""
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from test_two_tier_expected_raw_layout import seam

VALUES = [0, 17, 2**53 - 1, 2**53, 2**53 + 1, 2**53 + 3,
          2**64 - 1, np.int64(2**53 + 1), np.uint64(2**64 - 1)]

@pytest.mark.parametrize('seed', VALUES, ids=['zero','ordinary','below53','at53','above53','above53plus3','u64max','numpy-signed','numpy-unsigned'])
def test_seed_reaches_binding_exact(seam, monkeypatch, seed):
    module, unused = seam
    calls = []
    def binding(*args):
        calls.append(args)
        return dict(a_primary=[[1.2], [0.8]], a_specific=[0.5, 0.4],
                    threshold=[[0.2], [-0.2]], phi=[[1.]],
                    theta_p_eap=[[0.], [0.]], theta_p_sd=[[1.], [1.]],
                    category_counts=[[1, 1], [1, 1]], loglik_trace=[-3.],
                    n_iter=1, converged=False, termination_reason='max_iter_reached',
                    final_loglik_change=1., best_start=0, n_parameters=6,
                    primary_identification='correlated')
    monkeypatch.setattr(sys.modules[module.__package__+'.fitstats'], '_core_module',
                        lambda: SimpleNamespace(fit_two_tier_grm=binding))
    y = np.array([[0, 1], [1, 0]])
    pmap = np.ones((2, 1), dtype=bool)
    smap = np.zeros(2, dtype=np.int64)
    before = [a.copy() for a in [y, pmap, smap]]
    def call(value):
        return module.fit_two_tier_grm(y, pmap, smap, 2, 1, 1,
            q_primary=121, q_specific=241, max_iter=1, tol=1e-6,
            n_starts=2, seed=value)
    call(17)
    call(seed)
    assert len(calls) == 2 and unused == []
    assert type(calls[1][14]) is int and calls[1][14] == int(seed)
    for i, (a, b) in enumerate(zip(calls[0], calls[1])):
        if i == 14:
            continue
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        else:
            assert a == b
    for a, b in zip([y, pmap, smap], before):
        np.testing.assert_array_equal(a, b)

@pytest.mark.parametrize('seed', [True, -1, 2**64, 1.5, np.nan, np.inf])
def test_seed_invalid_still_denied(seam, seed):
    module, calls = seam
    with pytest.raises(ValueError, match='seed'):
        module._u64_seed(seed)
    assert calls == []

SCALARS = list(dict.fromkeys(getattr(np, n) for n in
    ('int8','int16','int32','int64','intp','longlong',
     'uint8','uint16','uint32','uint64','uintp','ulonglong')))
@pytest.mark.parametrize('scalar', SCALARS, ids=lambda s:s.__name__)
def test_seed_ordinary_numpy_types(seam, scalar):
    module, _ = seam
    assert module._u64_seed(scalar(17)) == 17

def test_seed_subclass_float_protocol_unchanged(seam):
    module, _ = seam
    seen = []
    class Existing(int):
        def __float__(self):
            seen.append('float')
            return 241.
        def __int__(self):
            raise AssertionError('new int callback not permitted')
    assert module._u64_seed(Existing(17)) == 241 and seen == ['float']
