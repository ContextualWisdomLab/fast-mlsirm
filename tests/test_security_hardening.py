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


# ---- VULN-0004 (2nd pass): non-finite / unbounded numeric config -----------
@pytest.mark.parametrize(
    "kw",
    [
        {"learning_rate": float("nan")},
        {"learning_rate": float("inf")},
        {"init_gamma": float("inf")},
        {"eps_distance": float("nan")},
        {"tolerance": float("nan")},
        {"gradient_clip": float("inf")},
        {"max_iter": 10**12},
        {"n_restarts": 10**12},
        {"m_steps": 10**12},
    ],
)
def test_config_rejects_nonfinite_or_unbounded_numerics(kw):
    with pytest.raises(ValueError):
        FitConfig(model="MLS2PLM", estimator="mmle", **kw).validate()


def test_config_accepts_normal_numerics():
    FitConfig(model="MLS2PLM", estimator="mmle", max_iter=100, n_restarts=2,
              m_steps=4, learning_rate=0.01, tolerance=1e-6,
              eps_distance=1e-8, init_gamma=1.0, gradient_clip=100.0).validate()


# ---- VULN-0005 (2nd pass): n_draws / serving_prior bounds -------------------
def test_serving_prior_rejects_extreme_n_dims():
    bundle = _bundle()
    bundle["n_dims"] = 2_147_483_647
    with pytest.raises(ValueError):
        serving.serving_prior(bundle)


def test_plausible_values_rejects_extreme_n_draws():
    if serving._core_module() is None:  # pragma: no cover - core built in CI
        pytest.skip("plausible_values requires the compiled Rust core")
    bundle = _bundle()
    for bad in (-1, 0, 10**20):
        with pytest.raises(ValueError):
            serving.plausible_values(bundle, {"q0": 1}, n_draws=bad)


# ---- VULN-0002 (confirm): malformed bundle -> ValueError, not KeyError ------
def test_score_respondents_rejects_bundle_missing_items():
    with pytest.raises(ValueError):
        serving.score_respondents({"schema_version": serving.SCHEMA_VERSION}, [{}])


# ===========================================================================
# Strix 2nd-batch VULN-0001..0011: preprocessing / inference / linking /
# validation / serving grid — input-validation & allocation-bound hardening.
# ===========================================================================
import types  # noqa: E402

from fast_mlsirm.inference import observed_information, oakes_standard_errors  # noqa: E402
from fast_mlsirm.linking import link_fixed_item_parameters  # noqa: E402
from fast_mlsirm.preprocessing import irtree_expand  # noqa: E402
from fast_mlsirm.types import MLSIRMParams  # noqa: E402
from fast_mlsirm.validation import _validate_labels  # noqa: E402


# ---- VULN-0001 (2nd): irtree_expand dense-allocation bound -----------------
def test_irtree_expand_rejects_oversized_expansion():
    y = np.zeros((1, 60_000))       # persons*items*nodes = 1*60000*900 = 5.4e7
    mapping = np.zeros((900, 2))
    with pytest.raises(ValueError, match="exceeds"):
        irtree_expand(y, mapping)


def test_irtree_expand_accepts_normal_shapes():
    y = np.array([[0.0, 1.0], [1.0, 0.0]])
    mapping = np.array([[0.0, 1.0], [1.0, 0.0]])  # 2 nodes x 2 cats
    expanded, factor_id = irtree_expand(y, mapping)
    assert expanded.shape == (2, 4) and factor_id.shape == (4,)


# ---- VULN-0002 (2nd): irtree node_dims must be finite non-negative ints ----
@pytest.mark.parametrize("bad", [np.array([0.5, 1.0]), np.array([-1.0, 0.0]),
                                 np.array([np.nan, 0.0]), np.array([np.inf, 0.0])])
def test_irtree_expand_rejects_bad_node_dims(bad):
    y = np.zeros((3, 2))
    mapping = np.zeros((2, 3))
    with pytest.raises(ValueError):
        irtree_expand(y, mapping, node_dims=bad)


# ---- VULN-0003 (2nd): label values above uint32 max ------------------------
def test_validate_labels_rejects_uint32_overflow():
    with pytest.raises(ValueError, match="uint32"):
        _validate_labels(np.array([0.0, 5_000_000_000.0]), "judge")


# ---- VULN-0011: human_human baseline length must match paired labels -------
def test_validate_judge_rejects_mismatched_human_a_length():
    judge = np.array([0, 1, 0, 1])
    human = np.array([0, 1, 1, 0])
    with pytest.raises(ValueError, match="length"):
        validate_judge(judge, human, k=2, human_human=(np.array([0, 1]),))


# ---- VULN-0004 (2nd): oakes factor_id validated before use -----------------
@pytest.mark.parametrize("bad", [np.array([0.0, np.nan, 1.0]),
                                 np.array([0.0, -1.0, 1.0]),
                                 np.array([0.5, 1.0, 2.0]),
                                 np.zeros((2, 3))])
def test_oakes_rejects_bad_factor_id(bad):
    result = types.SimpleNamespace(model="MLSRM", population={}, params=None)
    y = np.zeros((5, 3))
    with pytest.raises(ValueError):
        oakes_standard_errors(result, y, bad)


# ---- VULN-0005 (2nd): observed_information bounds the dense Hessian ---------
def test_observed_information_rejects_huge_parameter_vector():
    p = MLSIRMParams(theta=np.zeros((6000, 1)), alpha=np.zeros(1), b=np.zeros(1),
                     xi=np.zeros((1, 1)), zeta=np.zeros((1, 1)), tau=0.0)
    with pytest.raises(ValueError, match="at most"):
        observed_information(np.zeros((3, 1)), np.array([0]), p)


# ---- VULN-0006 (2nd): serving tensor-grid explosion ------------------------
def test_validate_bundle_rejects_grid_explosion():
    bundle = _bundle(latent_dim=4)
    bundle["model"] = "MLS2PLM"
    bundle["quadrature"] = {"q_theta": 21, "q_xi": 41}  # 41**4 = 2.8e6 > 1e6
    with pytest.raises(ValueError, match="grid limit"):
        serving._validate_bundle(bundle)


# ---- VULN-0007..0010: link_fixed_item_parameters anchor/param hardening -----
def _link_ns(alpha, b, theta):
    alpha = np.asarray(alpha, float)
    return types.SimpleNamespace(
        alpha=alpha, a=np.exp(alpha).reshape(-1, 1),
        b=np.asarray(b, float), theta=np.asarray(theta, float),
    )


def test_link_rejects_duplicate_anchors():        # VULN-0007
    s = _link_ns([0.0, 0.0], [0.0, 0.1], np.zeros((2, 1)))
    t = _link_ns([0.0, 0.0], [0.0, 0.1], np.zeros((2, 1)))
    with pytest.raises(ValueError, match="unique"):
        link_fixed_item_parameters(s, t, anchor_items=np.array([0, 0]))


def test_link_rejects_non_2d_theta():             # VULN-0008
    s = _link_ns([0.0, 0.0], [0.0, 0.1], np.zeros(2))
    t = _link_ns([0.0, 0.0], [0.0, 0.1], np.zeros(2))
    with pytest.raises(ValueError, match="2-D"):
        link_fixed_item_parameters(s, t, anchor_items=np.array([0, 1]))


@pytest.mark.parametrize("alpha", [[0.0, np.inf], [0.0, np.nan]])
def test_link_rejects_non_finite_params(alpha):   # VULN-0009
    s = _link_ns(alpha, [0.0, 0.1], np.zeros((2, 1)))
    t = _link_ns([0.0, 0.0], [0.0, 0.1], np.zeros((2, 1)))
    with pytest.raises(ValueError, match="finite"):
        link_fixed_item_parameters(s, t, anchor_items=np.array([0, 1]))


@pytest.mark.parametrize("anchors", [np.array([0.5, 1.0]), np.array([-1.0, 0.0]),
                                     np.array([np.nan, 0.0])])
def test_link_rejects_bad_anchor_indices(anchors):  # VULN-0010
    s = _link_ns([0.0, 0.0], [0.0, 0.1], np.zeros((2, 1)))
    t = _link_ns([0.0, 0.0], [0.0, 0.1], np.zeros((2, 1)))
    with pytest.raises(ValueError):
        link_fixed_item_parameters(s, t, anchor_items=anchors)
