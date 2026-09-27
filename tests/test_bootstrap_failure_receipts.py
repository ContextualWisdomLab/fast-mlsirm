"""Synthetic worker results exercise the actual bootstrap receipt path."""
import numpy as np
import pytest
import fast_mlsirm.bifactor_bootstrap as bb


def test_mixed_and_all_failed_receipts(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(bb, "fit_bifactor_grm", lambda **kw: SimpleNamespace(
        a_general=np.array([1.]), a_specific=np.array([0.]),
        threshold=np.array([[0.]]), loglik_trace=np.array([0.]),
        converged=False, termination_reason="max_iter_reached"))
    receipt = bb._fit_single_replicate(
        rep_idx=0, responses=np.zeros((2, 1)), specific_map=np.array([0]),
        n_cat=2, n_specific=1, group_ids=None, n_groups=1, anchor_mask=None,
        q_general=1, q_specific=1, max_iter=1, tol=1e-6, n_starts=1,
        rep_seed=0, estimate_specific_vars=False, device="cpu")
    assert receipt[1] is False
    assert receipt[9] == "fit not converged: max_iter_reached"
    failures = {1: "ValueError: category missing", 2: "fit not converged: max_iter_reached"}
    def worker(rep, *args):
        return (rep, rep not in failures, np.array([float(rep)]), np.array([0.]),
                np.array([[0.]]), np.array([0.]), np.array([1.]),
                np.ones((1, 1)), 0., failures.get(rep, ""))
    monkeypatch.setattr(bb, "_fit_single_replicate", worker)
    kw = dict(responses=np.zeros((2, 1)), specific_map=np.array([0]),
              n_cat=2, n_specific=1, n_replicates=3, batch_size=3,
              mc_stopping_ratio=0., compute_budget_seconds=60.,
              q_general=1, q_specific=1, base_seed=0, ci_level=.95,
              max_iter=1, n_starts=1, tol=1e-6, n_jobs=1)
    result = bb.run_bifactor_bootstrap(**kw)
    assert result.replicate_ids == (0, 1, 2)
    assert result.converged_replicate_ids == (0,)
    assert result.replicate_errors == ("", failures[1], failures[2])
    assert result.converged.tolist() == [True, False, False]
    assert result.replicate_a_general.tolist() == [[0.]]
    failures[0] = "RuntimeError: synthetic failure"
    with pytest.raises(RuntimeError, match="0/3") as raised:
        bb.run_bifactor_bootstrap(**kw)
    assert raised.value.replicate_ids == (0, 1, 2)
    assert raised.value.converged_replicate_ids == ()
    assert raised.value.replicate_errors == tuple(failures[i] for i in range(3))
    assert raised.value.converged.tolist() == [False, False, False]


def test_multigroup_real_worker_retains_flags_errors_and_parameter_row_ids(monkeypatch):
    from types import SimpleNamespace
    calls = []
    def fit(**kw):
        rep = len(calls)
        calls.append(kw)
        if rep == 2:
            raise ValueError("synthetic category missing")
        return SimpleNamespace(
            a_general=np.array([[10.], [20.]]), a_specific=np.zeros((2, 1)),
            threshold=np.zeros((2, 1, 1)), general_mean=np.array([0., 1.]),
            general_sd=np.ones(2), specific_sd=np.ones((2, 1)),
            loglik_trace=np.array([0.]), converged=rep == 0,
            termination_reason="tolerance_met" if rep == 0 else "max_iter_reached")
    monkeypatch.setattr(bb, "fit_bifactor_grm_multigroup", fit)
    result = bb.run_bifactor_bootstrap(
        responses=np.array([[0.], [1.], [0.], [1.]]), specific_map=np.array([0]),
        group_ids=np.array([0, 0, 1, 1]), n_groups=2,
        n_cat=2, n_specific=1, n_replicates=3, batch_size=3,
        mc_stopping_ratio=0., compute_budget_seconds=60.,
        q_general=1, q_specific=1, base_seed=0, ci_level=.95,
        max_iter=1, n_starts=1, tol=1e-6, n_jobs=1)
    assert result.n_groups == 2 and result.n_converged == 1
    assert result.replicate_ids == (0, 1, 2)
    assert result.converged_replicate_ids == (0,)
    assert result.converged.tolist() == [True, False, False]
    assert result.replicate_errors == (
        "", "fit not converged: max_iter_reached", "ValueError: synthetic category missing")
    assert result.replicate_a_general.tolist() == [[[10.], [20.]]]
    assert result.replicate_threshold.shape == (1, 2, 1, 1)
    assert len(calls) == 3
    assert all(call["group"].tolist() == [0, 0, 1, 1] for call in calls)
