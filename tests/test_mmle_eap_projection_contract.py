"""Regression evidence for the NumPy MMLE EAP matrix-vector projection."""

from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path

import numpy as np
from numpy.polynomial.hermite_e import hermegauss

from fast_mlsirm.estimators import mmle as mmle_module


ROOT = Path(__file__).resolve().parents[1]
CHANGELOG_FRAGMENT = ROOT / "docs" / "changelog.d" / "bolt-mmle-theta-optimization.md"


def _initial_posterior(
    responses: np.ndarray,
    observed: np.ndarray,
    *,
    n_nodes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct the first MMLE E-step without using production posterior helpers."""
    y_filled = np.where(observed, responses, 0.0)
    obs_f = observed.astype(np.float64)
    nodes, raw_weights = hermegauss(n_nodes)
    weights = raw_weights / raw_weights.sum()

    rng = np.random.default_rng(seed)
    p_item = (y_filled * obs_f).sum(0) / np.clip(obs_f.sum(0), 1.0, None)
    p_item = np.clip(p_item, 0.02, 0.98)
    discrimination = np.ones(responses.shape[1]) + 0.01 * rng.standard_normal(
        responses.shape[1]
    )
    intercept = np.log(p_item / (1.0 - p_item))

    logit = nodes[:, None] * discrimination[None, :] + intercept[None, :]
    log_p1 = -np.logaddexp(0.0, -logit)
    log_p0 = -np.logaddexp(0.0, logit)
    positive = (y_filled * obs_f) @ log_p1.T
    negative = ((1.0 - y_filled) * obs_f) @ log_p0.T
    log_joint = positive + negative + np.log(weights)[None, :]
    max_log_joint = log_joint.max(axis=1, keepdims=True)
    stabilized = np.exp(log_joint - max_log_joint)
    posterior = stabilized / stabilized.sum(axis=1, keepdims=True)
    return posterior, nodes


def test_eap_matvec_matches_previous_weighted_sum_under_partial_missingness() -> None:
    """The optimized EAP projection must preserve the previous weighted sum."""
    responses = np.array(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    observed = np.array(
        [
            [True, True, False, True],
            [True, False, True, True],
            [True, True, True, False],
            [False, True, True, True],
        ],
        dtype=bool,
    )
    n_nodes = 9
    seed = 17

    result = mmle_module.fit_mmle_2pl(
        responses,
        observed,
        n_nodes=n_nodes,
        max_iter=1,
        tol=0.0,
        seed=seed,
    )
    posterior, nodes = _initial_posterior(
        responses,
        observed,
        n_nodes=n_nodes,
        seed=seed,
    )
    previous_projection = (posterior * nodes[None, :]).sum(axis=1)

    np.testing.assert_allclose(
        result["theta"],
        previous_projection,
        rtol=1e-13,
        atol=1e-13,
    )


def test_eap_projection_source_uses_matrix_multiplication_without_broadcast_product() -> None:
    """Pin the allocation-bounded EAP source shape so the N-by-Q product stays absent."""
    source = textwrap.dedent(inspect.getsource(mmle_module.fit_mmle_2pl))
    tree = ast.parse(source)
    theta_assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "theta" for target in node.targets)
    ]

    assert len(theta_assignments) == 1
    theta_expression = theta_assignments[0].value
    assert isinstance(theta_expression, ast.BinOp)
    assert isinstance(theta_expression.op, ast.MatMult)
    assert isinstance(theta_expression.left, ast.Name)
    assert theta_expression.left.id == "posterior"
    assert isinstance(theta_expression.right, ast.Name)
    assert theta_expression.right.id == "nodes"


def test_changelog_avoids_universal_eap_speedup_claims() -> None:
    """Release notes must describe the optimization without a fixed speedup promise."""
    fragment = CHANGELOG_FRAGMENT.read_text(encoding="utf-8")
    lowered = fragment.casefold()

    assert "30x" not in lowered
    assert "runtime performance" in lowered
    assert "hardware" in lowered
    assert "blas" in lowered
