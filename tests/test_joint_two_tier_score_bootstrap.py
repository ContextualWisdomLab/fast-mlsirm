"""Shared-plan composition accounting; stub fits are not native acceptance."""
import hashlib
import os
import numpy as np
import pytest
from fast_mlsirm import two_tier_focal as module


def test_all_scales_share_slots_and_joint_failures_keep_actual_records(monkeypatch):
    """Check five-scale plan identity, ordered-key rejection and whole-replicate failure.

    Uses the existing empirical sampler's Efron1979 source contract and a stub
    producer to observe orchestration only; no numerical fitting is validated.
    """
    ids = np.arange(12)
    groups = np.tile([0, 1], 6)
    controls = dict(
        reference_group=0, n_cat=4, n_primary=2, n_specific=4, focal_primary=0,
        initial_mean=np.zeros((2, 6)), initial_sd=np.ones((2, 6)),
        fit_q_primary=7, fit_q_specific=7, score_q_primary=7, score_q_specific=7,
        q_nuisance=121, max_iter=2000, tol=1e-6, n_starts=2, seed=20260928,
        device="gpu", gpu_memory_budget_bytes=1 << 30, cache_item_tables=True,
    )
    scales = {str(i): dict(person_ids=ids.copy(), responses=np.column_stack([ids, np.full(12, i)]),
                          primary_map=np.ones((2, 2), bool), specific_map=np.array([0, 1]),
                          controls=controls) for i in range(5)}
    seen = []
    fail = [False]
    actual_fit = object()

    def producer(y, g, pmap, smap, **kwargs):
        """Record selected slots; optionally fail the final scale with a fit object."""
        seen.append((y[:, 0].copy(), g.copy(), kwargs["seed"]))
        if fail[0] and y[0, 1] == 4:
            error = RuntimeError("synthetic nonconverged group")
            error.group_id = 1
            error.fit = actual_fit
            raise error
        return {"rows": y[:, 0].copy()}

    from inspect import signature
    producer.__signature__ = signature(module.fit_score_two_tier_groups)
    monkeypatch.setattr(module, "fit_score_two_tier_groups", producer)
    args = dict(scales=scales, person_ids=ids, group_ids=groups,
                n_groups=2, n_replicates=2, base_seed=20260928)
    bad = scales | {"4": scales["4"] | {"person_ids": ids[::-1]}}
    with pytest.raises(ValueError, match="ordered person keys"):
        module.run_joint_two_tier_score_bootstrap(**(args | {"scales": bad}))
    assert not seen
    got = module.run_joint_two_tier_score_bootstrap(**args)
    assert set(got["successful_replicates"]) == {0, 1}
    assert len(seen) == 10
    for rep in range(2):
        for selected, g, seed in seen[rep * 5:(rep + 1) * 5]:
            np.testing.assert_array_equal(selected, got["bootstrap_indices"][rep])
            np.testing.assert_array_equal(g, groups)
        assert all(record["failure"] is None for record in got["records"])
    digest = hashlib.sha256(np.asarray(got["bootstrap_indices"].shape, dtype="<u8").tobytes())
    digest.update(memoryview(got["bootstrap_indices"]))
    assert digest.hexdigest() == got["bootstrap_indices_sha256"]
    assert not got["bootstrap_indices"].flags.writeable
    fail[0] = True
    failed = module.run_joint_two_tier_score_bootstrap(**args)
    assert not failed["successful_replicates"]
    assert failed["n_completed"] == 2
    for record in failed["records"]:
        assert record["failure"]["scale"] == "4"
        assert record["failure"]["fit"] is actual_fit
        assert record["failure"]["group_id"] == 1
        assert "synthetic nonconverged" in record["failure"]["traceback"]



@pytest.mark.skipif(os.environ.get("FOCAL_GPU_NATIVE") != "1",
                    reason="requires explicitly enabled actual GPU hardware")
def test_joint_five_scale_resample_uses_actual_gpu():
    """Exercise one shared resample through five real Cai2010 fit/score paths.

    Efron1979 resampling and Cai2010 numerical source contracts are inherited
    from the producers. Five synthetic six-latent response matrices use the
    existing continuous Gaussian fixture with different declared seeds. One
    replicate with 15/21 fit and score nodes tests execution, not uncertainty
    precision, study convergence or population recovery. No stub fitting.
    The first 7/7 run terminates loglik_decreased and is preserved separately;
    increased nodes are a sensitivity attempt, not a replacement replicate.
    """
    from fast_mlsirm.two_tier_focal import run_joint_two_tier_score_bootstrap
    from test_two_tier_focal_gaussian_recovery_native import _continuous_six_latent_fixture

    ids = np.arange(1024)
    groups = np.tile([0, 1], 512)
    controls = dict(
        reference_group=1, n_cat=4, n_primary=2, n_specific=4, focal_primary=0,
        initial_mean=np.zeros((2, 6)), initial_sd=np.ones((2, 6)),
        fit_q_primary=15, fit_q_specific=21, score_q_primary=15, score_q_specific=21,
        q_nuisance=121, max_iter=2000, tol=1e-6, n_starts=2, seed=20260928,
        device="gpu", gpu_memory_budget_bytes=1 << 30, cache_item_tables=True,
    )
    scales = {}
    for i in range(5):
        ref, ap, _, sm, _, _ = _continuous_six_latent_fixture(
            512, np.zeros(6), np.ones(6), 20260928 + 2 * i,
        )
        focal, _, _, _, _, _ = _continuous_six_latent_fixture(
            512, np.array([.30, -.25, .20, -.30, .35, -.15]),
            np.array([1.15, .85, .80, 1.20, 1.10, .90]), 20260929 + 2 * i,
        )
        y = np.stack([focal, ref], axis=1).reshape(1024, 16)
        scales[str(i)] = dict(person_ids=ids, responses=y, primary_map=ap != 0,
                              specific_map=sm, controls=controls)
    out = run_joint_two_tier_score_bootstrap(
        scales, ids, groups, n_groups=2, n_replicates=1, base_seed=20260928,
    )
    if out["records"][0]["failure"] is not None:
        failure = out["records"][0]["failure"]
        fit = failure["fit"]
        diagnostics = {key: getattr(fit, key, None) for key in (
            "n_iter", "termination_reason", "final_loglik_change", "loglik_trace",
            "latent_mean", "latent_sd", "q_primary", "q_specific", "tol",
        )}
        pytest.fail(f"scale={failure['scale']}, group={failure['group_id']}, "
                    f"actual_fit={diagnostics}\n{failure['traceback']}")
    assert set(out["successful_replicates"]) == {0}
    indices = out["bootstrap_indices"][0]
    np.testing.assert_array_equal(groups[indices], groups)
    for name, got in out["successful_replicates"][0].items():
        assert all(f.converged for f in got["group_fits"].values())
        assert all(s.backend == "gpu" for s in got["group_scores"].values())
        for group, rows in got["group_rows"].items():
            np.testing.assert_array_equal(rows, np.flatnonzero(groups == group))
        assert got["theta"].shape == (1024,)
        assert np.isfinite(got["expected_total"]).all()
        assert got["settings"]["score_q_primary"] == 15
