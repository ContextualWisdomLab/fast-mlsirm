"""Python surface of the opt-in lognormal ``|a|`` bifactor slope prior.

FORWARDING-ONLY evidence: a recording fake core stands in for Rust, so
nothing here exercises native MAP numerics. The numerical prior (MAP M-step,
multigroup common/anchored items, Oakes MAP curvature) is tested in Rust (``tests/unit/bifactor_grm_tests.rs`` and
``tests/unit/bifactor_oakes_tests.rs``). These tests pin the Python
contract with a recording fake core: validation before dispatch, exact
kwarg forwarding (``None`` stays ``None`` = unchanged MML call), and prior
provenance on every result.
"""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm import bifactor_grm, bifactor_multigroup, fitstats

N_ITEMS, N_CAT, N_SPECIFIC = 4, 3, 2
SMAP = np.array([0, 0, 1, 1])
RESPONSES = np.array(
    [[0, 1, 2, 0], [1, 2, 0, 1], [2, 0, 1, 2], [0, 2, 1, 0], [1, 1, 1, 1], [2, 2, 2, 2]]
)
N_PERSONS = RESPONSES.shape[0]
CONTROLS = dict(q_general=7, q_specific=7, max_iter=5, tol=1e-4, n_starts=1, seed=1)


class RecordingCore:
    def __init__(self) -> None:
        self.calls: dict[str, tuple] = {}

    def _fit(self, n_groups: int) -> dict:
        return {
            "a_general": np.ones(n_groups * N_ITEMS),
            "a_specific": np.ones(n_groups * N_ITEMS),
            "threshold": np.tile([0.5, -0.5], n_groups * N_ITEMS),
            "theta_g_eap": np.zeros(N_PERSONS),
            "theta_g_sd": np.ones(N_PERSONS),
            "category_counts": np.ones(N_ITEMS * N_CAT, dtype=np.int64),
            "group_category_counts": np.ones(n_groups * N_ITEMS * N_CAT, dtype=np.int64),
            "general_mean": np.zeros(n_groups),
            "general_sd": np.ones(n_groups),
            "specific_sd": np.ones(n_groups * N_SPECIFIC),
            "loglik_trace": np.array([-1.0]),
            "em_objective_trace": np.array([-1.5]),
            "n_iter": 1,
            "converged": True,
            "termination_reason": "tolerance_met",
            "final_loglik_change": 0.0,
            "best_start": 0,
            "n_parameters": 1,
        }

    def _with_prior(self, res: dict, args: tuple) -> dict:
        # The Rust core echoes the fitted prior; provenance is read from it.
        res["slope_prior_mu"], res["slope_prior_sd"] = args[-2:]
        return res

    def fit_bifactor_grm(self, *args):
        self.calls["fit_bifactor_grm"] = args
        return self._with_prior(self._fit(1), args)

    def fit_bifactor_grm_multigroup(self, *args):
        self.calls["fit_bifactor_grm_multigroup"] = args
        return self._with_prior(self._fit(args[3]), args)

    def bifactor_oakes_se(self, *args):
        self.calls["bifactor_oakes_se"] = args
        return {
            "labels": ["a_general:0"],
            "information": [[1.0]],
            "vcov": [[1.0]],
            "se": [1.0],
            "positive_definite": True,
            "non_pd_reason": None,
        }


@pytest.fixture
def core(monkeypatch: pytest.MonkeyPatch) -> RecordingCore:
    fake = RecordingCore()
    monkeypatch.setattr(fitstats, "_core_module", lambda: fake)
    return fake


def _fit(**prior):
    return bifactor_grm.fit_bifactor_grm(RESPONSES, SMAP, N_CAT, N_SPECIFIC, **CONTROLS, **prior)


def _fit_mg(**prior):
    group = np.arange(N_PERSONS) % 2
    return bifactor_multigroup.fit_bifactor_grm_multigroup(
        RESPONSES, group, SMAP, N_CAT, N_SPECIFIC, **CONTROLS, **prior
    )


def _oakes(**prior):
    return bifactor_grm.bifactor_oakes_se(
        np.ones(N_ITEMS),
        np.ones(N_ITEMS),
        np.tile([0.5, -0.5], (N_ITEMS, 1)),
        RESPONSES,
        SMAP,
        N_CAT,
        N_SPECIFIC,
        7,
        7,
        1e-5,
        **prior,
    )


@pytest.mark.parametrize(
    ("call", "core_name"),
    [
        (_fit, "fit_bifactor_grm"),
        (_fit_mg, "fit_bifactor_grm_multigroup"),
        (_oakes, "bifactor_oakes_se"),
    ],
)
def test_prior_kwargs_forward_and_are_recorded(core, call, core_name):
    mml = call()
    assert core.calls[core_name][-2:] == (None, None)
    assert (mml.slope_prior_mu, mml.slope_prior_sd) == (None, None)

    mu = float(np.log(0.8))
    map_ = call(slope_prior_mu=np.float64(mu), slope_prior_sd=0.5)
    forwarded = core.calls[core_name][-2:]
    assert forwarded == (mu, 0.5)
    assert all(type(v) is float for v in forwarded)
    assert (map_.slope_prior_mu, map_.slope_prior_sd) == (mu, 0.5)
    if core_name != "bifactor_oakes_se":
        np.testing.assert_array_equal(map_.em_objective_trace, [-1.5])


@pytest.mark.parametrize("call", [_fit, _fit_mg, _oakes])
@pytest.mark.parametrize(
    ("prior", "message"),
    [
        ({"slope_prior_mu": 0.0}, "provided together"),
        ({"slope_prior_sd": 1.0}, "provided together"),
        ({"slope_prior_mu": float("nan"), "slope_prior_sd": 1.0}, "slope_prior_mu"),
        ({"slope_prior_mu": 0.0, "slope_prior_sd": 0.0}, "slope_prior_sd"),
        ({"slope_prior_mu": 0.0, "slope_prior_sd": float("inf")}, "slope_prior_sd"),
        ({"slope_prior_mu": True, "slope_prior_sd": 1.0}, "slope_prior_mu"),
        ({"slope_prior_mu": np.bool_(True), "slope_prior_sd": 1.0}, "slope_prior_mu"),
        ({"slope_prior_mu": 0.0, "slope_prior_sd": np.bool_(True)}, "slope_prior_sd"),
        ({"slope_prior_mu": np.complex128(0.1 + 3j), "slope_prior_sd": 1.0}, "slope_prior_mu"),
        ({"slope_prior_mu": 0.0, "slope_prior_sd": 0.5 + 0j}, "slope_prior_sd"),
        ({"slope_prior_mu": np.array(0.1), "slope_prior_sd": 1.0}, "slope_prior_mu"),
        ({"slope_prior_mu": "x", "slope_prior_sd": 1.0}, "real numbers"),
        ({"slope_prior_mu": object(), "slope_prior_sd": 1.0}, "real numbers"),
    ],
)
def test_invalid_prior_is_rejected_before_core(core, call, prior, message):
    with pytest.raises(ValueError, match=message):
        call(**prior)
    assert core.calls == {}


@pytest.mark.parametrize("prior", [{}, {"slope_prior_mu": 0.0, "slope_prior_sd": 0.5}])
def test_multigroup_oakes_fails_closed_before_core(core, prior):
    mg = _fit_mg(**prior)
    core.calls.clear()
    for slopes in (mg, mg.a_general):
        with pytest.raises(ValueError, match="joint item and group mean/variance information"):
            bifactor_grm.bifactor_oakes_se(
                slopes, mg.a_specific, mg.threshold, RESPONSES, SMAP,
                N_CAT, N_SPECIFIC, 7, 7, 1e-5,
                slope_prior_mu=mg.slope_prior_mu,
                slope_prior_sd=mg.slope_prior_sd,
            )
    assert core.calls == {}


@pytest.mark.parametrize("prior", [{}, {"slope_prior_mu": 0.0, "slope_prior_sd": 0.5}])
def test_oakes_from_fit_rejects_selected_multigroup_row(core, prior):
    mg = _fit_mg(**prior)
    with pytest.raises((TypeError, ValueError), match="#2113"):
        bifactor_grm.bifactor_oakes_se_from_fit(
            mg, RESPONSES[::2], q_general=7, q_specific=7, fd_step=1e-5
        )
    assert "bifactor_oakes_se" not in core.calls


@pytest.mark.parametrize("prior", [{}, {"slope_prior_mu": 0.0, "slope_prior_sd": 0.5}])
def test_oakes_from_fit_matches_raw_call_with_fitted_prior(core, prior):
    fit = _fit(**prior)
    from_fit = bifactor_grm.bifactor_oakes_se_from_fit(
        fit, RESPONSES, q_general=7, q_specific=7, fd_step=1e-5
    )
    from_fit_args = core.calls["bifactor_oakes_se"]
    raw = bifactor_grm.bifactor_oakes_se(
        fit.a_general, fit.a_specific, fit.threshold, RESPONSES, SMAP,
        fit.n_cat, fit.n_specific, 7, 7, 1e-5,
        slope_prior_mu=fit.slope_prior_mu, slope_prior_sd=fit.slope_prior_sd,
    )
    raw_args = core.calls["bifactor_oakes_se"]
    for left, right in zip(from_fit_args, raw_args):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(from_fit.information, raw.information)
    np.testing.assert_array_equal(from_fit.se, raw.se)
    assert (from_fit.slope_prior_mu, from_fit.slope_prior_sd) == raw_args[-2:]


def test_oakes_from_fit_validates_response_shape_and_categories(core):
    fit = _fit()
    for responses in (RESPONSES[:, :-1], np.full_like(RESPONSES, N_CAT)):
        with pytest.raises(ValueError, match="responses"):
            bifactor_grm.bifactor_oakes_se_from_fit(
                fit, responses, q_general=7, q_specific=7, fd_step=1e-5
            )
    assert "bifactor_oakes_se" not in core.calls


def test_raw_oakes_fixed_fixture_remains_unchanged(core):
    result = _oakes()
    assert result.labels == ["a_general:0"]
    np.testing.assert_array_equal(result.information, [[1.0]])
    np.testing.assert_array_equal(result.se, [1.0])
    assert core.calls["bifactor_oakes_se"][-2:] == (None, None)
