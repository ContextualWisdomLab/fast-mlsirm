"""Reference-prior admission at Python dispatch; no fitting is performed."""
from types import SimpleNamespace

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
import fast_mlsirm.two_tier_fipc as scorer
from tests.test_two_tier_fipc import ANCHOR, SPECIFIC_MAP, _fit


@pytest.mark.parametrize("shape", ["scalar", "vector"])
@pytest.mark.parametrize("field,value", [
    ("primary_mean", np.nan), ("primary_mean", np.inf),
    ("primary_mean", -np.inf), ("primary_sd", np.nan),
    ("primary_sd", np.inf), ("primary_sd", -np.inf),
    ("primary_sd", 0.0), ("primary_sd", -1.0),
])
def test_invalid_reference_prior_is_rejected_before_native_dispatch(
    monkeypatch, shape, field, value,
):
    """Reject malformed priors before native acquisition on the shared route."""
    calls = []

    def acquire():
        calls.append("acquire")
        return SimpleNamespace(two_tier_grm_reference_score_moments=lambda *a: {
            "mean": 1.0, "second_moment": 2.0, "variance": 1.0,
        })

    monkeypatch.setattr(fitstats, "_core_module", acquire)
    bad = value if shape == "scalar" else np.array([0.5, value])
    with pytest.raises(ValueError, match=field):
        scorer.two_tier_reference_expected_score_moments(
            _fit(), SPECIFIC_MAP, ANCHOR, q_primary=121, q_specific=121,
            **{field: bad},
        )
    assert calls == []


def test_finite_nonunit_reference_prior_preserves_native_arguments(monkeypatch):
    """A recording-only core proves valid caller priors are neither dropped nor reset."""
    calls = []

    def moments(*args):
        calls.append(args)
        return {"mean": 1.0, "second_moment": 2.0, "variance": 1.0}

    monkeypatch.setattr(fitstats, "_core_module", lambda: SimpleNamespace(
        two_tier_grm_reference_score_moments=moments,
    ))
    mean = np.array([-0.3, 0.4])
    sd = np.array([0.8, 1.2])
    result = scorer.two_tier_reference_expected_score_moments(
        _fit(), SPECIFIC_MAP, ANCHOR, primary_mean=mean, primary_sd=sd,
        q_primary=121, q_specific=241,
    )
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][7], mean)
    np.testing.assert_array_equal(calls[0][8], sd)
    assert calls[0][9:] == (121, 241)
    np.testing.assert_array_equal(result.primary_mean, mean)
    np.testing.assert_array_equal(result.primary_sd, sd)
