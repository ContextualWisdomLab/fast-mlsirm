"""Two-tier reference-metric person-score contracts.

The Rust reference moments are checked against an independent NumPy oracle
(explicit category probabilities integrated on a NumPy product grid) and the
scorer is exercised on a real ``fit_two_tier_grm`` fit.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest
from fast_mlsirm.two_tier_fipc import (
    execute_two_tier_fipc_group_person_score_payload,
    score_two_tier_fipc_group_persons,
    two_tier_reference_expected_score_moments,
)
from fast_mlsirm.two_tier_grm import expected_raw_two_tier_grm, fit_two_tier_grm


@dataclass(frozen=True)
class _Fit:
    a_primary: np.ndarray
    a_specific: np.ndarray
    threshold: np.ndarray
    phi: np.ndarray
    theta_p_eap: np.ndarray
    theta_p_sd: np.ndarray
    n_primary: int = 2
    n_specific: int = 2
    n_cat: int = 4
    primary_identification: str = "orthogonal"


SPECIFIC_MAP = np.array([0, 0, 1, 1, 0, -1], dtype=np.int64)
ANCHOR = np.array([True, True, False, False, True, False])


def _fit() -> _Fit:
    return _Fit(
        a_primary=np.array(
            [[1.2, 0.0], [1.0, 0.0], [0.9, 0.6], [0.8, 0.5], [0.0, 1.1], [0.0, 0.9]]
        ),
        a_specific=np.array([0.7, 0.5, 0.6, -0.4, 0.8, 0.0]),
        threshold=np.tile(np.array([1.4, 0.0, -1.3]), (6, 1)),
        phi=np.eye(2),
        theta_p_eap=np.array([[-1.2, 0.3], [0.0, 0.0], [0.9, -1.4]]),
        theta_p_sd=np.full((3, 2), 0.5),
    )


def _gh(q: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.hermite_e.hermegauss(q)
    return nodes, weights / weights.sum()


def _oracle_moments(fit, smap, anchor, p_mean, p_sd, q_p, q_s):
    """Anchor-only E[T | theta] moments from explicit GRM category probabilities."""
    nodes, weights = _gh(q_p)
    s_nodes, s_weights = _gh(q_s)
    mesh = np.meshgrid(*[p_mean[d] + p_sd[d] * nodes for d in range(p_mean.size)], indexing="ij")
    grid = np.stack([m.reshape(-1) for m in mesh], axis=1)
    w = np.prod([m.reshape(-1) for m in np.meshgrid(*[weights] * p_mean.size, indexing="ij")], axis=0)
    totals = np.zeros(grid.shape[0])
    for i in np.flatnonzero(anchor):
        base = grid @ fit.a_primary[i]
        if smap[i] >= 0 and fit.a_specific[i] != 0.0:
            base = base[:, None] + fit.a_specific[i] * s_nodes[None, :]
        cum = 1.0 / (1.0 + np.exp(-(base[..., None] + fit.threshold[i])))
        ones = np.ones(base.shape + (1,))
        probs = np.concatenate((ones - cum[..., :1], -np.diff(cum, axis=-1), cum[..., -1:]), axis=-1)
        e = probs @ np.arange(probs.shape[-1])
        totals += e @ s_weights if e.ndim == 2 else e
    mean, second = float(w @ totals), float(w @ totals**2)
    return mean, second, second - mean * mean


def test_reference_moments_match_numpy_oracle() -> None:
    fit = _fit()
    p_mean, p_sd = np.array([0.4, -0.3]), np.array([1.2, 0.8])
    got = two_tier_reference_expected_score_moments(
        fit, SPECIFIC_MAP, ANCHOR, primary_mean=p_mean, primary_sd=p_sd, q_primary=9, q_specific=11
    )
    mean, second, variance = _oracle_moments(fit, SPECIFIC_MAP, ANCHOR, p_mean, p_sd, 9, 11)
    assert got.mean == pytest.approx(mean, abs=1e-10)
    assert got.second_moment == pytest.approx(second, abs=1e-9)
    assert got.variance == pytest.approx(variance, abs=1e-9)
    assert got.n_anchor_items == int(ANCHOR.sum())


def test_person_scores_reuse_fit_eaps_and_shared_expected_raw() -> None:
    fit = _fit()
    result = score_two_tier_fipc_group_persons(fit, SPECIFIC_MAP, ANCHOR, q_primary=7, q_specific=9)
    np.testing.assert_array_equal(result.theta_primary_eap, fit.theta_p_eap)
    np.testing.assert_array_equal(result.theta_primary_sd, fit.theta_p_sd)
    np.testing.assert_array_equal(result.expected_raw, expected_raw_two_tier_grm(fit, SPECIFIC_MAP, 9))


def test_reference_moments_ignore_non_anchor_rows() -> None:
    fit = _fit()
    baseline = two_tier_reference_expected_score_moments(fit, SPECIFIC_MAP, ANCHOR, q_primary=7, q_specific=9)
    ap, asp, thr = fit.a_primary.copy(), fit.a_specific.copy(), fit.threshold.copy()
    ap[~ANCHOR], asp[~ANCHOR], thr[~ANCHOR] = 8.0, -7.0, np.array([4.0, 0.0, -4.0])
    poisoned = two_tier_reference_expected_score_moments(
        replace(fit, a_primary=ap, a_specific=asp, threshold=thr), SPECIFIC_MAP, ANCHOR,
        q_primary=7, q_specific=9,
    )
    assert poisoned.mean == baseline.mean
    assert poisoned.variance == baseline.variance


def test_scores_a_real_orthogonal_two_tier_fit() -> None:
    rng = np.random.default_rng(20260929)
    n_persons, n_items = 120, 6
    primary_map = np.array([[True, False]] * 3 + [[False, True]] * 3)
    specific_map = np.array([0, 0, 1, 1, 0, 1], dtype=np.int64)
    theta = rng.standard_normal((n_persons, 2))
    spec = rng.standard_normal((n_persons, 2))
    eta = (theta @ (primary_map * 1.2).T) + 0.6 * spec[:, specific_map]
    y = (eta[..., None] + np.array([1.0, -1.0]) > rng.logistic(size=(n_persons, n_items, 1))).sum(-1)
    fit = fit_two_tier_grm(
        y, primary_map, specific_map, n_cat=3, n_primary=2, n_specific=2, q_primary=7,
        q_specific=7, max_iter=40, tol=1e-4, n_starts=1, seed=0, primary_correlation="identity",
    )
    anchor = np.array([True, False, True, True, False, True])
    result = score_two_tier_fipc_group_persons(fit, specific_map, anchor, q_primary=7, q_specific=7)
    assert result.expected_raw.shape == (n_persons,)
    assert np.all((result.expected_raw >= 0.0) & (result.expected_raw <= 2.0 * n_items))
    assert 0.0 < result.reference_moments.mean < 2.0 * int(anchor.sum())
    np.testing.assert_array_equal(result.theta_primary_eap, fit.theta_p_eap)


def test_correlated_fit_fails_closed() -> None:
    with pytest.raises(ValueError, match="orthogonal"):
        score_two_tier_fipc_group_persons(
            replace(_fit(), primary_identification="correlated"), SPECIFIC_MAP, ANCHOR,
            q_primary=5, q_specific=5,
        )
    with pytest.raises(ValueError, match="Phi == I"):
        score_two_tier_fipc_group_persons(
            replace(_fit(), phi=np.array([[1.0, 0.2], [0.2, 1.0]])), SPECIFIC_MAP, ANCHOR,
            q_primary=5, q_specific=5,
        )


def test_malformed_inputs_fail_closed() -> None:
    fit = _fit()
    with pytest.raises(ValueError, match="anchor"):
        score_two_tier_fipc_group_persons(fit, SPECIFIC_MAP, np.zeros(6, bool), q_primary=5, q_specific=5)
    with pytest.raises(TypeError, match="q_primary"):
        score_two_tier_fipc_group_persons(fit, SPECIFIC_MAP, ANCHOR, q_primary=5.0, q_specific=5)
    with pytest.raises(ValueError, match="sd"):
        score_two_tier_fipc_group_persons(
            fit, SPECIFIC_MAP, ANCHOR, q_primary=5, q_specific=5, reference_primary_sd=0.0
        )
    with pytest.raises(ValueError, match="strictly decreasing"):
        score_two_tier_fipc_group_persons(
            replace(fit, threshold=np.tile(np.array([0.0, 0.5, -1.0]), (6, 1))), SPECIFIC_MAP,
            ANCHOR, q_primary=5, q_specific=5,
        )
    with pytest.raises(ValueError, match="specific_map"):
        score_two_tier_fipc_group_persons(
            fit, np.array([0, 0, 1, 1, 0.5, -1]), ANCHOR, q_primary=5, q_specific=5
        )


def test_payload_contract_is_serializable_and_explicit() -> None:
    fit = _fit()
    fields = ("a_primary", "a_specific", "threshold", "phi", "theta_p_eap", "theta_p_sd")
    payload = {
        "fit": {name: np.asarray(getattr(fit, name)).tolist() for name in fields}
        | {"n_primary": 2, "n_specific": 2, "n_cat": 4, "primary_identification": "orthogonal"},
        "specific_map": SPECIFIC_MAP.tolist(),
        "anchor": ANCHOR.tolist(),
        "q_primary": 5,
        "q_specific": 5,
    }
    out = execute_two_tier_fipc_group_person_score_payload(payload)
    direct = score_two_tier_fipc_group_persons(fit, SPECIFIC_MAP, ANCHOR, q_primary=5, q_specific=5)
    assert out["family"] == "two_tier_fipc_group_person_score"
    assert out["model_scope"] == "two_tier_grm_orthogonal_primary"
    assert out["expected_raw"] == direct.expected_raw.tolist()
    with pytest.raises(ValueError, match="missing"):
        execute_two_tier_fipc_group_person_score_payload({"fit": {}})


# Posterior SD is supplied fit data, not recomputed by these admission tests.
# Recording callbacks verify pre-dispatch rejection, not numerical accuracy.
@pytest.mark.parametrize("route", ["scorer", "payload"])
@pytest.mark.parametrize(
    "case", ["scalar", "flat", "wrong_rows", "wrong_columns", "nan", "positive_inf",
             "negative_inf", "negative", "none"],
)
def test_posterior_sd_rejected_before_scoring_dispatch(monkeypatch, route, case):
    import fast_mlsirm.two_tier_fipc as scorer

    values = {
        "scalar": 0.5,
        "flat": np.full(6, 0.5),
        "wrong_rows": np.full((2, 2), 0.5),
        "wrong_columns": np.full((3, 1), 0.5),
        "nan": np.full((3, 2), np.nan),
        "positive_inf": np.full((3, 2), np.inf),
        "negative_inf": np.full((3, 2), -np.inf),
        "negative": np.full((3, 2), -0.5),
        "none": None,
    }
    fit = replace(_fit(), theta_p_sd=values[case])
    calls = []

    def reference(*args, **kwargs):
        calls.append("reference")
        return scorer.TwoTierReferenceExpectedScoreMoments(
            3.0, 10.0, 1.0, np.zeros(2), np.ones(2), 3, 121, 241
        )

    def expected(*args, **kwargs):
        calls.append("expected_raw")
        return np.array([2.0, 3.0, 4.0])

    monkeypatch.setattr(scorer, "two_tier_reference_expected_score_moments", reference)
    monkeypatch.setattr(scorer, "expected_raw_two_tier_grm", expected)
    with pytest.raises(ValueError, match="theta_p_sd"):
        if route == "scorer":
            scorer.score_two_tier_fipc_group_persons(
                fit, SPECIFIC_MAP, ANCHOR, q_primary=121, q_specific=241
            )
        else:
            scorer.execute_two_tier_fipc_group_person_score_payload({
                "fit": vars(fit), "specific_map": SPECIFIC_MAP.tolist(),
                "anchor": ANCHOR.tolist(), "q_primary": 121, "q_specific": 241,
            })
    assert calls == [], "invalid posterior SD must fail before scoring dispatch"


@pytest.mark.parametrize("route", ["scorer", "payload"])
@pytest.mark.parametrize("case", ["ordinary", "zero", "noncontiguous"])
def test_posterior_sd_valid_values_preserve_scoring_transport(monkeypatch, route, case):
    import json
    import fast_mlsirm.two_tier_fipc as scorer

    sd = np.full((3, 2), 0.5)
    if case == "zero":
        sd.fill(0.0)
    elif case == "noncontiguous":
        sd = np.full((3, 4), 0.5)[:, ::2]
        assert not sd.flags.c_contiguous
    fit = replace(_fit(), theta_p_sd=sd)
    before = {name: np.asarray(getattr(fit, name)).copy() for name in (
        "a_primary", "a_specific", "threshold", "phi", "theta_p_eap", "theta_p_sd"
    )}
    calls = []
    expected_values = np.array([2.0, 3.0, 4.0])

    def reference(got_fit, smap, anchor, **kwargs):
        np.testing.assert_array_equal(got_fit.theta_p_eap, fit.theta_p_eap)
        np.testing.assert_array_equal(smap, SPECIFIC_MAP)
        np.testing.assert_array_equal(anchor, ANCHOR)
        assert kwargs["q_primary"] == 121 and kwargs["q_specific"] == 241
        calls.append("reference")
        return scorer.TwoTierReferenceExpectedScoreMoments(
            3.0, 10.0, 1.0, np.zeros(2), np.ones(2), 3, 121, 241
        )

    def expected(got_fit, smap, q_specific):
        np.testing.assert_array_equal(got_fit.theta_p_eap, fit.theta_p_eap)
        np.testing.assert_array_equal(smap, SPECIFIC_MAP)
        assert q_specific == 241
        calls.append("expected_raw")
        return expected_values

    monkeypatch.setattr(scorer, "two_tier_reference_expected_score_moments", reference)
    monkeypatch.setattr(scorer, "expected_raw_two_tier_grm", expected)
    if route == "scorer":
        output = scorer.score_two_tier_fipc_group_persons(
            fit, SPECIFIC_MAP, ANCHOR, q_primary=121, q_specific=241
        )
        np.testing.assert_array_equal(output.theta_primary_eap, fit.theta_p_eap)
        np.testing.assert_array_equal(output.theta_primary_sd, sd)
        np.testing.assert_array_equal(output.expected_raw, expected_values)
    else:
        output = scorer.execute_two_tier_fipc_group_person_score_payload({
            "fit": vars(fit), "specific_map": SPECIFIC_MAP.tolist(),
            "anchor": ANCHOR.tolist(), "q_primary": 121, "q_specific": 241,
        })
        assert output["theta_primary_eap"] == fit.theta_p_eap.tolist()
        assert output["theta_primary_sd"] == sd.tolist()
        assert output["expected_raw"] == expected_values.tolist()
        json.dumps(output, allow_nan=False)
    assert calls == ["reference", "expected_raw"]
    for name, values in before.items():
        np.testing.assert_array_equal(getattr(fit, name), values)


@pytest.mark.parametrize("route", ["scorer", "payload"])
def test_posterior_sd_missing_field_keeps_predispatch_rejection(monkeypatch, route):
    from types import SimpleNamespace
    import fast_mlsirm.two_tier_fipc as scorer

    fields = vars(_fit()).copy()
    del fields["theta_p_sd"]
    calls = []

    def forbidden(*args, **kwargs):
        calls.append("scoring_dispatch")
        raise AssertionError("missing posterior SD must not dispatch scoring")

    # Keep the genuine reference helper's existing required-field check.
    monkeypatch.setattr(scorer, "expected_raw_two_tier_grm", forbidden)
    with pytest.raises(TypeError, match="theta_p_sd"):
        if route == "scorer":
            scorer.score_two_tier_fipc_group_persons(
                SimpleNamespace(**fields), SPECIFIC_MAP, ANCHOR,
                q_primary=121, q_specific=241,
            )
        else:
            scorer.execute_two_tier_fipc_group_person_score_payload({
                "fit": fields, "specific_map": SPECIFIC_MAP.tolist(),
                "anchor": ANCHOR.tolist(), "q_primary": 121, "q_specific": 241,
            })
    assert calls == []


# Native-return admission controls: explicit recorded native values, not fit truth.
@pytest.mark.parametrize("route", ["reference", "scorer", "payload"])
@pytest.mark.parametrize("field", ["mean", "second_moment", "variance"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")], ids=["nan", "posinf", "neginf"])
def test_reference_moments_nonfinite_rejected_before_expected_scoring(monkeypatch, route, field, bad):
    from types import SimpleNamespace
    import fast_mlsirm.fitstats as fitstats
    import fast_mlsirm.two_tier_fipc as scorer

    fit = _fit()
    saved = {name: np.asarray(getattr(fit, name)).copy() for name in (
        "a_primary", "a_specific", "threshold", "phi", "theta_p_eap", "theta_p_sd"
    )}
    native = {"mean": 3.0, "second_moment": 10.0, "variance": 1.0}
    native[field] = bad
    calls = []

    def reference(*args):
        calls.append("reference")
        assert args[-2:] == (121, 241)
        return dict(native)

    def expected(*args, **kwargs):
        calls.append("expected_raw")
        return np.array([2.0, 3.0, 4.0])

    monkeypatch.setattr(fitstats, "_core_module", lambda: SimpleNamespace(
        two_tier_grm_reference_score_moments=reference
    ))
    monkeypatch.setattr(scorer, "expected_raw_two_tier_grm", expected)
    try:
        with pytest.raises(ValueError, match="reference"):
            if route == "reference":
                scorer.two_tier_reference_expected_score_moments(
                    fit, SPECIFIC_MAP, ANCHOR, q_primary=121, q_specific=241
                )
            elif route == "scorer":
                scorer.score_two_tier_fipc_group_persons(
                    fit, SPECIFIC_MAP, ANCHOR, q_primary=121, q_specific=241
                )
            else:
                scorer.execute_two_tier_fipc_group_person_score_payload({
                    "fit": vars(fit), "specific_map": SPECIFIC_MAP.tolist(),
                    "anchor": ANCHOR.tolist(), "q_primary": 121, "q_specific": 241,
                })
        # Repaired contract: reference transport occurs, expected transport does not.
        assert calls == ["reference"]
    finally:
        for name, value in saved.items():
            np.testing.assert_array_equal(getattr(fit, name), value)


@pytest.mark.parametrize("route", ["reference", "scorer", "payload"])
@pytest.mark.parametrize("case", ["ordinary", "zero_variance"])
def test_reference_moments_finite_values_preserve_transport(monkeypatch, route, case):
    import json
    from types import SimpleNamespace
    import fast_mlsirm.fitstats as fitstats
    import fast_mlsirm.two_tier_fipc as scorer

    fit = _fit()
    native = {"mean": 3.0, "second_moment": 10.0, "variance": 1.0}
    if case == "zero_variance":
        native.update(second_moment=9.0, variance=0.0)
    calls = []

    def reference(*args):
        calls.append("reference")
        assert args[-2:] == (121, 241)
        return dict(native)

    def expected(got_fit, smap, q):
        calls.append("expected_raw")
        assert got_fit is fit or np.array_equal(got_fit.theta_p_eap, fit.theta_p_eap)
        np.testing.assert_array_equal(smap, SPECIFIC_MAP)
        assert q == 241
        return np.array([2.0, 3.0, 4.0])

    monkeypatch.setattr(fitstats, "_core_module", lambda: SimpleNamespace(
        two_tier_grm_reference_score_moments=reference
    ))
    monkeypatch.setattr(scorer, "expected_raw_two_tier_grm", expected)
    if route == "reference":
        result = scorer.two_tier_reference_expected_score_moments(
            fit, SPECIFIC_MAP, ANCHOR, q_primary=121, q_specific=241
        )
        actual = {name: getattr(result, name) for name in native}
        assert result.q_primary == 121 and result.q_specific == 241
    elif route == "scorer":
        result = scorer.score_two_tier_fipc_group_persons(
            fit, SPECIFIC_MAP, ANCHOR, q_primary=121, q_specific=241
        )
        actual = {name: getattr(result.reference_moments, name) for name in native}
        np.testing.assert_array_equal(result.expected_raw, [2.0, 3.0, 4.0])
    else:
        result = scorer.execute_two_tier_fipc_group_person_score_payload({
            "fit": vars(fit), "specific_map": SPECIFIC_MAP.tolist(),
            "anchor": ANCHOR.tolist(), "q_primary": 121, "q_specific": 241,
        })
        actual = {name: result["reference_expected_score"][name] for name in native}
        assert result["expected_raw"] == [2.0, 3.0, 4.0]
        json.dumps(result, allow_nan=False)
    assert actual == native
    assert calls == (["reference"] if route == "reference" else ["reference", "expected_raw"])
    json.dumps(actual, allow_nan=False)


@pytest.mark.parametrize("route", ["direct", "scorer", "payload"])
@pytest.mark.parametrize("values", [
    2.0, [], [1.0, 2.0], [1.0, 2.0, 3.0, 4.0],
    [[1.0, 2.0, 3.0]], [[1.0], [2.0], [3.0]], [[[1.0, 2.0, 3.0]]],
], ids=["scalar", "empty", "short", "long", "row", "column", "rank3"])
def test_expected_raw_shape_rejected_on_nine_argument_routes(monkeypatch, route, values):
    """Require one score per input row; a local transport contract, not a model change."""
    import fast_mlsirm.fitstats as fitstats
    from types import SimpleNamespace
    fit = _fit()
    returned = np.asarray(values, dtype=np.float64)
    calls = []
    before = {name: getattr(fit, name).copy() for name in
              ("a_primary", "a_specific", "threshold", "theta_p_eap", "theta_p_sd")}

    def expected(*args):
        assert len(args) == 9 and args[5:] == (4, 2, 2, 241)
        calls.append("expected")
        return returned

    def reference(*args):
        assert args[-2:] == (121, 241)
        calls.append("reference")
        return {"mean": 1.0, "second_moment": 2.0, "variance": 1.0}

    monkeypatch.setattr(fitstats, "_core_module", lambda: SimpleNamespace(
        two_tier_expected_raw=expected, two_tier_grm_reference_score_moments=reference))
    try:
        with pytest.raises(ValueError, match="expected raw scores must have shape"):
            if route == "direct":
                expected_raw_two_tier_grm(fit, SPECIFIC_MAP, 241)
            elif route == "scorer":
                score_two_tier_fipc_group_persons(fit, SPECIFIC_MAP, ANCHOR,
                                                 q_primary=121, q_specific=241)
            else:
                execute_two_tier_fipc_group_person_score_payload({
                    "fit": vars(fit), "specific_map": SPECIFIC_MAP.tolist(),
                    "anchor": ANCHOR.tolist(), "q_primary": 121, "q_specific": 241})
    finally:
        assert calls == (["expected"] if route == "direct" else ["reference", "expected"])
        for name, value in before.items():
            np.testing.assert_array_equal(getattr(fit, name), value)
        np.testing.assert_array_equal(returned, np.asarray(values, dtype=np.float64))


@pytest.mark.parametrize("route", ["direct", "scorer", "payload"])
@pytest.mark.parametrize("rows", [0, 1, 3])
def test_expected_raw_shape_valid_vectors_preserve_nine_argument_routes(monkeypatch, route, rows):
    """Keep empty, singleton and multirow scores without flattening or padding."""
    import fast_mlsirm.fitstats as fitstats
    from types import SimpleNamespace
    import json
    fit = replace(_fit(), theta_p_eap=_fit().theta_p_eap[:rows].copy(),
                  theta_p_sd=_fit().theta_p_sd[:rows].copy())
    returned = np.array([1.0, 2.0, 3.0])[:rows].copy()
    calls = []

    def expected(*args):
        assert len(args) == 9 and args[5:] == (4, 2, 2, 241)
        calls.append("expected")
        return returned

    def reference(*args):
        assert args[-2:] == (121, 241)
        calls.append("reference")
        return {"mean": 1.0, "second_moment": 2.0, "variance": 1.0}

    monkeypatch.setattr(fitstats, "_core_module", lambda: SimpleNamespace(
        two_tier_expected_raw=expected, two_tier_grm_reference_score_moments=reference))
    if route == "direct":
        actual = expected_raw_two_tier_grm(fit, SPECIFIC_MAP, 241)
    elif route == "scorer":
        actual = score_two_tier_fipc_group_persons(fit, SPECIFIC_MAP, ANCHOR,
                                                  q_primary=121, q_specific=241).expected_raw
    else:
        result = execute_two_tier_fipc_group_person_score_payload({
            "fit": vars(fit), "specific_map": SPECIFIC_MAP.tolist(),
            "anchor": ANCHOR.tolist(), "q_primary": 121, "q_specific": 241})
        json.dumps(result, allow_nan=False)
        actual = np.asarray(result["expected_raw"], dtype=np.float64)
    assert actual.shape == (rows,)
    np.testing.assert_array_equal(actual, returned)
    assert calls == (["expected"] if route == "direct" else ["reference", "expected"])
