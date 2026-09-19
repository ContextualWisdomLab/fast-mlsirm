"""Fail-closed length checks for the two_tier_grm_marginal_loglik PyO3 export."""

from __future__ import annotations

import numpy as np
import pytest

from tests.test_cai2010_two_tier_grm_gates import _g2_fixture


def _marginal_loglik_core():
    try:
        from fast_mlsirm import _core
    except Exception:  # pragma: no cover
        pytest.skip("compiled core not available")
    if not hasattr(_core, "two_tier_grm_marginal_loglik"):  # pragma: no cover
        pytest.skip("core built without two_tier_grm_marginal_loglik")
    return _core.two_tier_grm_marginal_loglik


def _valid_call_kwargs() -> dict:
    y, a_p, a_s, thr, phi, pmap, smap = _g2_fixture()
    n_persons, n_items = y.shape
    n_primary = a_p.shape[1]
    n_specific = int(np.max(smap) + 1)
    n_cat = thr.shape[1] + 1
    return {
        "a_primary": a_p.reshape(-1),
        "a_specific": a_s.reshape(-1),
        "threshold": thr.reshape(-1),
        "phi": phi.reshape(-1),
        "y": y.astype(np.int64).reshape(-1),
        "observed": np.ones(n_persons * n_items, dtype=bool),
        "primary_map": pmap.reshape(-1),
        "specific_map": smap.reshape(-1),
        "n_persons": n_persons,
        "n_items": n_items,
        "n_primary": n_primary,
        "n_specific": n_specific,
        "n_cat": n_cat,
        "q_primary": 5,
        "q_specific": 5,
    }


def _call(core, kwargs: dict) -> float:
    return float(
        core(
            kwargs["a_primary"],
            kwargs["a_specific"],
            kwargs["threshold"],
            kwargs["phi"],
            kwargs["y"],
            kwargs["observed"],
            kwargs["primary_map"],
            kwargs["specific_map"],
            kwargs["n_persons"],
            kwargs["n_items"],
            kwargs["n_primary"],
            kwargs["n_specific"],
            kwargs["n_cat"],
            kwargs["q_primary"],
            kwargs["q_specific"],
        )
    )


def test_valid_inputs_return_finite_loglik() -> None:
    """Sized inputs still evaluate after boundary validation."""
    core = _marginal_loglik_core()
    loglik = _call(core, _valid_call_kwargs())
    assert np.isfinite(loglik)


def test_short_observed_mask_with_negative_y_raises_value_error() -> None:
    """Short observed + negative y at an unmasked index must not panic in Rust."""
    core = _marginal_loglik_core()
    kwargs = _valid_call_kwargs()
    n_cells = kwargs["n_persons"] * kwargs["n_items"]
    kwargs["y"] = kwargs["y"].copy()
    kwargs["y"][0] = -1
    kwargs["observed"] = np.ones(n_cells - 1, dtype=bool)

    with pytest.raises(ValueError, match="observed must have length n_persons \\* n_items"):
        _call(core, kwargs)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("y", "y must have length n_persons \\* n_items"),
        ("observed", "observed must have length n_persons \\* n_items"),
        ("primary_map", "primary_map must have length n_items \\* n_primary"),
        ("specific_map", "specific_map must have length n_items"),
        ("a_primary", "a_primary must have length n_items \\* n_primary"),
        ("a_specific", "a_specific must have length n_items"),
        ("threshold", "thresholds must have length n_items \\* \\(n_cat - 1\\)"),
        ("phi", "phi must have length n_primary \\* n_primary"),
    ],
)
def test_marginal_loglik_rejects_length_mismatch(field: str, message: str) -> None:
    """Flattened cardinalities are validated before indexing or copying."""
    core = _marginal_loglik_core()
    kwargs = _valid_call_kwargs()
    kwargs[field] = kwargs[field][:-1]

    with pytest.raises(ValueError, match=message):
        _call(core, kwargs)
