"""Python routing tests with explicit synthetic native receipts, not fit evidence."""

from types import SimpleNamespace

import numpy as np
import pytest

import fast_mlsirm.two_tier_focal as api


def inputs():
    return dict(
        responses=np.array([[0, 1, 2, 0], [1, 2, 0, 1], [np.nan, -1, 2, 0]]),
        primary_map=np.ones((4, 1), dtype=bool),
        specific_map=np.zeros(4, dtype=np.int64),
        a_primary=np.array([[0.8], [1.2], [0.6], [1.1]]),
        a_specific=np.array([0.4, -0.3, 0.7, -0.2]),
        threshold=np.tile([0.8, -0.6], (4, 1)),
        latent_mean=np.array([0.3, -0.2]),
        latent_sd=np.array([1.1, 0.8]),
        n_cat=3,
        n_primary=1,
        n_specific=1,
        q_primary=7,
        q_specific=7,
        max_iter=1,
        tol=1e-6,
    )


def test_native_argument_order_shapes_and_nonconvergence_record(monkeypatch):
    calls = []

    def fit(*args):
        calls.append(args)
        return dict(
            person_mean=list(range(6)),
            person_second=list(range(10, 16)),
            person_sd=[0.5] * 6,
            loglik=-123.0,
            latent_mean=[9.0, 8.0],
            latent_sd=[2.0, 3.0],
            loglik_trace=[-100.0, -123.0],
            n_iter=1,
            converged=False,
            termination_reason="loglik_decreased",
            final_loglik_change=-23.0,
            initial_mean=args[7],
            initial_sd=args[8],
            q_primary=args[14],
            q_specific=args[15],
            max_iter=args[16],
            tol=args[17],
        )

    def score(*args):
        calls.append(args)
        return dict(
            person_mean=list(range(6)),
            person_second=list(range(10, 16)),
            person_sd=[0.5] * 6,
            loglik=-123.0,
        )

    monkeypatch.setattr(
        api,
        "_core_module",
        lambda: SimpleNamespace(
            fit_two_tier_grm_focal_orthogonal=fit, score_two_tier_grm_orthogonal=score
        ),
    )
    data = inputs()
    result = api.fit_two_tier_grm_focal_orthogonal(**data)
    assert result.scores.mean.shape == (3, 2)
    np.testing.assert_array_equal(result.scores.mean, np.arange(6).reshape(3, 2))
    assert not result.converged
    assert result.termination_reason == "loglik_decreased"
    assert result.final_loglik_change == -23.0
    assert result.scores.loglik == -123.0
    np.testing.assert_array_equal(result.latent_mean, [9.0, 8.0])
    np.testing.assert_array_equal(calls[0][0], [0, 1, 2, 0, 1, 2, 0, 1, 0, 0, 2, 0])
    np.testing.assert_array_equal(calls[0][1], [True] * 8 + [False, False, True, True])
    assert calls[0][9:] == (3, 4, 1, 1, 3, 7, 7, 1, 1e-6)
    data.pop("max_iter")
    data.pop("tol")
    scored = api.score_two_tier_grm_orthogonal(**data)
    assert scored.second.shape == (3, 2)
    assert scored.loglik == -123.0
    assert len(calls) == 2


@pytest.mark.parametrize(
    "key,value",
    [
        ("n_primary", True),
        ("q_primary", 0),
        ("q_specific", 1.5),
        ("max_iter", 0),
        ("tol", float("inf")),
        ("specific_map", np.array([0.5, 0, 0, 0])),
        ("specific_map", np.array([2**64 - 1, 0, 0, 0], dtype=np.uint64)),
        ("primary_map", np.ones((4, 1), dtype=int)),
        ("responses", np.array([[0, 1, 2, float("inf")]])),
        ("responses", np.array([[0, 1, 2, -2]])),
        ("responses", np.array([[0, 1, 2, 0.5]])),
        ("responses", np.array([[0, 1, 2, 3]])),
        ("responses", np.array([[0, 1, 2, 1j]])),
        ("latent_sd", np.array([1.0, 0.0])),
        ("latent_mean", np.array([float("nan"), 0])),
        ("threshold", np.tile([0.0, 0.0], (4, 1))),
    ],
)
def test_invalid_input_rejected_before_native_dispatch(monkeypatch, key, value):
    def forbidden():
        pytest.fail("invalid input reached native loader")

    monkeypatch.setattr(api, "_core_module", forbidden)
    data = inputs()
    data[key] = value
    with pytest.raises(ValueError):
        api.fit_two_tier_grm_focal_orthogonal(**data)


def test_missing_core_and_scalar_callbacks_are_not_fallbacks(monkeypatch):
    monkeypatch.setattr(api, "_core_module", lambda: None)
    with pytest.raises(RuntimeError, match="updated compiled Rust core"):
        api.fit_two_tier_grm_focal_orthogonal(**inputs())

    class Trap:
        def __int__(self):
            pytest.fail("caller int callback invoked")

        def __float__(self):
            pytest.fail("caller float callback invoked")

    data = inputs()
    data["max_iter"] = Trap()
    with pytest.raises(ValueError):
        api.fit_two_tier_grm_focal_orthogonal(**data)
