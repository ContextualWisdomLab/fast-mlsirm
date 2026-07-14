"""Regression tests for the Strix VULN-0001..0007 input-validation hardening:
untrusted population labels, judge labels, fit-config sizes, serving-bundle
structure/finiteness, and plausible-values response domain."""

from __future__ import annotations

import json

import numpy as np
import pytest

from fast_mlsirm import serving
from fast_mlsirm.config import MAX_LATENT_DIM, MAX_XI_POINTS, FitConfig
from fast_mlsirm.fit import _compact_population_labels
from fast_mlsirm.validation import validate_judge


# ---- VULN-0001: sparse/invalid population labels ---------------------------
def test_population_labels_compacted_not_unbounded():
    ids, n = _compact_population_labels(np.array([0, 999_999_999]), 2, "group_id")
    assert n == 2 and ids.tolist() == [0, 1]  # bounded by distinct count, not max+1
    # already-contiguous labels are unchanged
    ids2, n2 = _compact_population_labels(np.array([0, 1, 1, 2, 0]), 5, "cluster_id")
    assert n2 == 3 and ids2.tolist() == [0, 1, 1, 2, 0]


@pytest.mark.parametrize(
    "bad",
    [np.array([-1, 0]), np.array([0.5, 1.5]), np.array([np.nan, 1.0]), np.array([[0], [1]])],
)
def test_population_labels_reject_invalid(bad):
    with pytest.raises(ValueError):
        _compact_population_labels(bad, bad.shape[0] if bad.ndim == 1 else 2, "group_id")


# ---- VULN-0004 / VULN-0005: unbounded config sizes -------------------------
def test_config_rejects_extreme_latent_dim():
    with pytest.raises(ValueError):
        FitConfig(model="MLS2PLM", latent_dim=1_000_000_000).validate()
    FitConfig(model="MLS2PLM", latent_dim=MAX_LATENT_DIM).validate()  # boundary ok


def test_config_rejects_extreme_xi_points():
    with pytest.raises(ValueError):
        FitConfig(model="MLS2PLM", xi_rule="qmc", xi_points=100_000_000).validate()
    FitConfig(model="MLS2PLM", xi_rule="qmc", xi_points=MAX_XI_POINTS).validate()


# ---- VULN-0003: judge label coercion ---------------------------------------
@pytest.mark.parametrize(
    "labels",
    [np.array([0.9, 1.9]), np.array([-1.0, 0.0]), np.array([np.nan, 1.0]),
     np.array([np.inf, 1.0]), np.array([0, 5]), np.array([[0], [1]])],
)
def test_validate_judge_rejects_bad_labels(labels):
    with pytest.raises(ValueError):
        validate_judge(labels, np.array([0, 1]), k=2)


# ---- serving-bundle helpers ------------------------------------------------
def _bundle(n_items=1, n_dims=1, latent_dim=1):
    return {
        "schema_version": serving.SCHEMA_VERSION,
        "model": "MIRT",
        "n_items": n_items,
        "n_dims": n_dims,
        "latent_dim": latent_dim,
        "quadrature": {"q_theta": 7, "q_xi": 7},
        "eps_distance": 1e-8,
        "tau": -30.0,
        "population": None,
        "items": [
            {"code": f"q{j}", "factor_id": 0, "alpha": 0.0, "b": 0.0, "zeta": [0.0] * latent_dim}
            for j in range(n_items)
        ],
    }


# ---- VULN-0006: non-finite bundle JSON / params ----------------------------
def test_load_bundle_rejects_nonfinite_json(tmp_path):
    p = tmp_path / "b.json"
    bundle = _bundle()
    bundle["tau"] = float("inf")
    p.write_text(json.dumps(bundle, allow_nan=True), encoding="utf-8")
    with pytest.raises(ValueError):
        serving.load_serving_bundle(p)


def test_score_respondents_rejects_nonfinite_params():
    bundle = _bundle()
    bundle["items"][0]["alpha"] = float("nan")
    with pytest.raises(ValueError):
        serving.score_respondents(bundle, {"q0": 1})


# ---- VULN-0007: oversized / inconsistent bundle dimensions -----------------
def test_score_respondents_rejects_oversized_n_items():
    bundle = _bundle()
    bundle["n_items"] = 10**12
    bundle["items"] = []
    with pytest.raises(ValueError):
        serving.score_respondents(bundle, [{}])


def test_score_respondents_rejects_out_of_range_factor_id():
    bundle = _bundle()
    bundle["items"][0]["factor_id"] = 99  # n_dims == 1
    with pytest.raises(ValueError):
        serving.score_respondents(bundle, {"q0": 1})


def test_score_respondents_rejects_item_count_mismatch():
    bundle = _bundle(n_items=2)
    bundle["items"] = bundle["items"][:1]  # len(items) != n_items
    with pytest.raises(ValueError):
        serving.score_respondents(bundle, {"q0": 1})


# ---- VULN-0002: plausible_values non-binary/non-finite responses -----------
def test_plausible_values_rejects_non_binary_response():
    if serving._core_module() is None:  # pragma: no cover - core is built in CI
        pytest.skip("plausible_values requires the compiled Rust core")
    bundle = _bundle()
    for bad in (2.0, float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            serving.plausible_values(bundle, {"q0": bad}, n_draws=2)
