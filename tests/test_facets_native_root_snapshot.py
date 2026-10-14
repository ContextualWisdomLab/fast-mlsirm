"""Root-snapshot evidence for the many-facet Rust result bridge."""

from __future__ import annotations

import numpy as np

import fast_mlsirm.facets as facets


def test_native_fit_result_root_rebinding_cannot_change_admitted_metadata(
    monkeypatch,
) -> None:
    """Later native-owned root rebinding cannot redefine admitted fit evidence."""

    payload: dict[str, object] = {
        "item_difficulty": np.array([-0.25, 0.25], dtype=np.float64),
        "rater_severity": np.array([0.0], dtype=np.float64),
        "thresholds": np.array([0.0], dtype=np.float64),
        "theta": np.array([-0.5, 0.5], dtype=np.float64),
        "loglik_trace": np.array([-5.0], dtype=np.float64),
        "n_iter": 1,
        "converged": True,
        "connected": True,
        "n_parameters": 2,
    }
    real_array = facets.np.array
    rebound = False

    def array_and_rebind(value: object, *args: object, **kwargs: object) -> np.ndarray:
        nonlocal rebound
        if not rebound:
            rebound = True
            payload["n_iter"] = 5
            payload["converged"] = False
            payload["connected"] = False
            payload["n_parameters"] = 999
        return real_array(value, *args, **kwargs)

    monkeypatch.setattr(facets.np, "array", array_and_rebind)

    result = facets._validate_native_fit_result(
        payload,
        n_persons=2,
        n_items=2,
        n_raters=1,
        n_cat=2,
        max_iter=5,
    )

    assert rebound is True
    assert result[5:] == (1, True, True, 2)
