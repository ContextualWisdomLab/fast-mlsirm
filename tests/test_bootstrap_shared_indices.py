"""One person-row plan must drive both model refits through the real worker."""
from types import SimpleNamespace
import hashlib
import struct
import numpy as np
import pytest
import fast_mlsirm.bifactor_bootstrap as bb


def arguments():
    return dict(responses=np.arange(4.).reshape(4, 1), specific_map=np.array([0]),
                n_cat=2, n_specific=1, n_replicates=2, batch_size=2,
                mc_stopping_ratio=0., compute_budget_seconds=60.,
                q_general=1, q_specific=1, base_seed=0, ci_level=.95,
                max_iter=1, n_starts=1, tol=1e-6, n_jobs=2)


@pytest.mark.parametrize('n_groups', [1, 2])
def test_two_models_use_identical_rows_and_canonical_plan_hash(monkeypatch, n_groups):
    calls = []
    plan = np.array([[1, 1, 3, 2], [0, 1, 2, 2]], dtype=np.int64)
    def fit(**kw):
        calls.append(kw)
        shape = (1,) if n_groups == 1 else (2, 1)
        return SimpleNamespace(a_general=np.ones(shape), a_specific=np.zeros(shape),
            threshold=np.zeros((*shape, 1)), loglik_trace=np.array([0.]),
            general_mean=np.zeros(n_groups), general_sd=np.ones(n_groups),
            specific_sd=np.ones((n_groups, 1)), converged=True)
    monkeypatch.setattr(bb, 'fit_bifactor_grm', fit)
    monkeypatch.setattr(bb, 'fit_bifactor_grm_multigroup', fit)
    def forbidden(*args):
        raise AssertionError('supplied person plan was replaced by a new draw')
    monkeypatch.setattr(bb, '_generate_bootstrap_indices', forbidden)
    kw = arguments()
    if n_groups == 2:
        kw.update(group_ids=np.array([0, 0, 1, 1]), n_groups=2)
    first = bb.run_bifactor_bootstrap(**kw, bootstrap_indices=plan)
    first_calls = sorted(calls, key=lambda c: c['seed'])
    calls.clear()
    kw['responses'] = kw['responses'] + 10.
    second = bb.run_bifactor_bootstrap(**kw, bootstrap_indices=plan.astype('>i8'))
    second_calls = sorted(calls, key=lambda c: c['seed'])
    for rep in range(2):
        assert first_calls[rep]['responses'].ravel().tolist() == plan[rep].tolist()
        assert (second_calls[rep]['responses'].ravel() - 10.).tolist() == plan[rep].tolist()
    expected = hashlib.sha256(struct.pack('<QQ', 2, 4) + struct.pack('<8q', *plan.ravel())).hexdigest()
    assert first.bootstrap_indices_sha256 == second.bootstrap_indices_sha256 == expected
    assert first.replicate_ids == second.replicate_ids == (0, 1)


@pytest.mark.parametrize('plan', [
    np.zeros(4, dtype=int), np.zeros((2, 4), dtype=float),
    np.zeros((2, 4), dtype=bool), np.full((2, 4), -1), np.full((2, 4), 4),
    np.full((2, 4), 2**64-1, dtype=np.uint64),
    np.array([[0, 2, 1, 3], [0, 1, 2, 3]]),
])
def test_bad_plan_rejected_before_dispatch(monkeypatch, plan):
    def forbidden(*args):
        raise AssertionError('bad plan reached worker dispatch')
    monkeypatch.setattr(bb, '_fit_single_replicate', forbidden)
    with pytest.raises(ValueError, match='bootstrap_indices'):
        bb.run_bifactor_bootstrap(**arguments(), group_ids=np.array([0, 0, 1, 1]),
                                 n_groups=2, bootstrap_indices=plan)


def test_plan_snapshot_survives_caller_mutation_and_all_failed_run(monkeypatch):
    plan = np.array([[1, 1, 3, 2], [0, 1, 2, 2]], dtype=np.int64)
    expected_rows = plan.tolist()
    expected_hash = hashlib.sha256(struct.pack('<QQ', 2, 4) + struct.pack('<8q', *plan.ravel())).hexdigest()
    seen = []
    def fit(**kw):
        seen.append(kw['responses'].ravel().tolist())
        plan.fill(-1)
        raise ValueError('synthetic failure')
    monkeypatch.setattr(bb, 'fit_bifactor_grm', fit)
    kw = arguments()
    kw['n_jobs'] = 1
    with pytest.raises(RuntimeError, match='0/2') as raised:
        bb.run_bifactor_bootstrap(**kw, bootstrap_indices=plan)
    assert seen == expected_rows
    assert raised.value.bootstrap_indices_sha256 == expected_hash
    assert raised.value.replicate_ids == (0, 1)
    assert raised.value.replicate_errors == ('ValueError: synthetic failure',) * 2


@pytest.mark.parametrize('leaf', ['bifactor_grm', 'bifactor_multigroup', 'two_tier_grm', 'grm'])
@pytest.mark.parametrize('seed', [2**53+1, 2**64-1, np.uint64(2**64-1)])
def test_fit_seed_admission_preserves_exact_integer_bits(leaf, seed):
    import importlib
    module = importlib.import_module('fast_mlsirm.' + leaf)
    assert module._u64_seed(seed) == int(seed)


@pytest.mark.parametrize('n_groups', [1, 2])
def test_public_plan_matches_internal_draws_through_real_workers(monkeypatch, n_groups):
    kw = arguments()
    kw.update(base_seed=2**64-1, n_jobs=1)
    if n_groups == 2:
        kw.update(group_ids=np.array([0, 0, 1, 1]), n_groups=2)
    seen = []
    def fit(**args):
        seen.append(args['responses'].ravel().tolist())
        raise ValueError('synthetic outcome')
    monkeypatch.setattr(bb, 'fit_bifactor_grm', fit)
    monkeypatch.setattr(bb, 'fit_bifactor_grm_multigroup', fit)
    plan = bb.generate_person_bootstrap_indices(4, 2, base_seed=kw['base_seed'],
        group_ids=kw.get('group_ids'), n_groups=n_groups)
    with pytest.raises(RuntimeError, match='0/2'):
        bb.run_bifactor_bootstrap(**kw)
    assert seen == plan.tolist()
    seen.clear()
    with pytest.raises(RuntimeError, match='0/2') as supplied:
        bb.run_bifactor_bootstrap(**kw, bootstrap_indices=plan)
    assert seen == plan.tolist()
    assert supplied.value.bootstrap_indices_sha256 is not None


@pytest.mark.parametrize('seed', [True, np.bool_(True), 1., '1', -1, 2**64])
def test_public_generator_rejects_invalid_seed_before_sampling(monkeypatch, seed):
    def forbidden(*args):
        raise AssertionError('invalid seed reached sampling')
    monkeypatch.setattr(bb, '_generate_bootstrap_indices', forbidden)
    with pytest.raises(ValueError, match='base_seed'):
        bb.generate_person_bootstrap_indices(4, 2, base_seed=seed)


def test_public_generator_uses_existing_group_admission(monkeypatch):
    def forbidden(*args):
        raise AssertionError('invalid strata reached public sampler')
    monkeypatch.setattr(bb, '_generate_bootstrap_indices', forbidden)
    with pytest.raises(ValueError, match='group'):
        bb.generate_person_bootstrap_indices(4, 2, base_seed=0,
            group_ids=np.array([0., 0., .5, 1.]), n_groups=2)


def test_seed_admission_does_not_call_integer_subclass_conversion():
    from fast_mlsirm._seed import _u64_seed
    class Trap(int):
        def __int__(self):
            raise AssertionError('caller conversion reached')
        def __float__(self):
            raise AssertionError('caller conversion reached')
    with pytest.raises(ValueError, match='seed'):
        _u64_seed(Trap(1))
