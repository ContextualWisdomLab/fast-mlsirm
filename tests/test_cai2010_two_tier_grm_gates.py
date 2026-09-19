"""Paper-rewritten NumPy gates for Cai (2010) two-tier GRM (#2040).

Independent oracles derived from the journal text; crate likelihood helpers
are not used inside the oracle paths (G1 compares *to* ``grm_logprobs`` only).

References (APA 7th ed.)
------------------------
Cai, L. (2010). A two-tier full-information item factor analysis model with
applications. *Psychometrika, 75*(4), 581-612.
https://doi.org/10.1007/s11336-010-9178-0
Journal pp. 586 (eq. 1), 589 (eq. 11-12), 589-590 (eq. 15).
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.polynomial.hermite_e import hermegauss

from fast_mlsirm.two_tier_grm import encode_specific_map_from_columns, fit_two_tier_grm


def _gh(q: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = hermegauss(q)
    return nodes, weights / weights.sum()


def _cai2010_grm_category_logprobs(
    base: float,
    alpha: np.ndarray,
) -> np.ndarray:
    """Cai (2010) eq. (11)-(12), p. 589, in log space.

    ``P+(k|eta,xi) = logistic(alpha_k + base)`` for ``k = 1..K-1``,
    ``P+(0)=1``, ``P+(K)=0``, and ``P_k = P+(k) - P+(k+1)``.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    kb = alpha.size
    if kb < 1:
        raise ValueError("alpha must have length K-1 >= 1")
    pplus = np.empty(kb + 2, dtype=np.float64)
    pplus[0] = 1.0
    eta = alpha + base
    pplus[1:-1] = 1.0 / (1.0 + np.exp(-eta))
    pplus[-1] = 0.0
    pk = pplus[:-1] - pplus[1:]
    if np.any(pk < 0.0):
        return np.full(kb + 1, np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(pk)


def _log_sum_exp(xs: np.ndarray) -> float:
    mx = float(np.max(xs))
    if not np.isfinite(mx):
        return float("-inf")
    return mx + float(np.log(np.sum(np.exp(xs - mx))))


def _primary_grid_nodes_weights(q: int, n_primary: int) -> tuple[np.ndarray, np.ndarray]:
    tz, wz = _gh(q)
    n_grid = q**n_primary
    coords = np.zeros((n_grid, n_primary), dtype=np.float64)
    log_w0 = np.zeros(n_grid, dtype=np.float64)
    log_w = np.log(wz)
    for g in range(n_grid):
        tail = g
        acc = 0.0
        for d in range(n_primary):
            digit = tail % q
            tail //= q
            coords[g, d] = tz[digit]
            acc += log_w[digit]
        log_w0[g] = acc
    return coords, log_w0


def _reweighted_primary_log_weights(
    log_w0: np.ndarray,
    coords: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    p = phi.shape[0]
    phi_inv = np.linalg.inv(phi)
    logdet = float(np.linalg.slogdet(phi)[1])
    out = np.empty(log_w0.shape[0], dtype=np.float64)
    for g, lw0 in enumerate(log_w0):
        z = coords[g]
        quad = float(z @ (phi_inv - np.eye(p)) @ z)
        out[g] = lw0 - 0.5 * (logdet + quad)
    return out


def _item_base(
    a_primary_row: np.ndarray,
    a_specific: float,
    theta_p: np.ndarray,
    theta_s: float,
) -> float:
    return float(a_primary_row @ theta_p + a_specific * theta_s)


def _cai2010_marginal_loglik_unreduced(
    y: np.ndarray,
    a_primary: np.ndarray,
    a_specific: np.ndarray,
    thresholds: np.ndarray,
    phi: np.ndarray,
    primary_map: np.ndarray,
    specific_map: np.ndarray,
    q_primary: int,
    q_specific: int,
) -> float:
    """Cai (2010) eq. (15) first line: (p+S)-fold product-grid quadrature."""
    n_persons, n_items = y.shape
    n_primary = a_primary.shape[1]
    n_specific = int(np.max(specific_map) + 1)
    coords_p, log_w0 = _primary_grid_nodes_weights(q_primary, n_primary)
    log_w_p = _reweighted_primary_log_weights(log_w0, coords_p, phi)
    ts, ws = _gh(q_specific)
    log_ws = np.log(ws)
    n_gp = coords_p.shape[0]
    n_gs = ts.shape[0]
    blocks: list[list[int]] = [[] for _ in range(n_specific)]
    for i, s in enumerate(specific_map):
        blocks[int(s)].append(i)
    loglik = 0.0
    for p in range(n_persons):
        log_nodes = np.empty(n_gp * (n_gs**n_specific), dtype=np.float64)
        idx = 0
        for gp in range(n_gp):
            theta_p = coords_p[gp]
            for tail in range(n_gs**n_specific):
                acc = log_w_p[gp]
                spec_coords = np.empty(n_specific, dtype=np.float64)
                tmp = tail
                for s in range(n_specific):
                    digit = tmp % n_gs
                    tmp //= n_gs
                    spec_coords[s] = ts[digit]
                    acc += log_ws[digit]
                for i in range(n_items):
                    row_mask = primary_map[i]
                    base = _item_base(
                        a_primary[i, row_mask],
                        float(a_specific[i]),
                        theta_p[row_mask],
                        float(spec_coords[int(specific_map[i])]),
                    )
                    lp = _cai2010_grm_category_logprobs(base, thresholds[i])
                    acc += lp[int(y[p, i])]
                log_nodes[idx] = acc
                idx += 1
        loglik += _log_sum_exp(log_nodes)
    return loglik


def _cai2010_marginal_loglik_reduced(
    y: np.ndarray,
    a_primary: np.ndarray,
    a_specific: np.ndarray,
    thresholds: np.ndarray,
    phi: np.ndarray,
    primary_map: np.ndarray,
    specific_map: np.ndarray,
    q_primary: int,
    q_specific: int,
) -> float:
    """Cai (2010) eq. (15) second line: factored specific integrals (p+1 dims)."""
    n_persons, n_items = y.shape
    n_primary = a_primary.shape[1]
    n_specific = int(np.max(specific_map) + 1)
    coords_p, log_w0 = _primary_grid_nodes_weights(q_primary, n_primary)
    log_w_p = _reweighted_primary_log_weights(log_w0, coords_p, phi)
    ts, ws = _gh(q_specific)
    log_ws = np.log(ws)
    n_gp = coords_p.shape[0]
    qs = ts.shape[0]
    blocks: list[list[int]] = [[] for _ in range(n_specific)]
    for i, s in enumerate(specific_map):
        blocks[int(s)].append(i)
    loglik = 0.0
    for p in range(n_persons):
        log_like_g = np.empty(n_gp, dtype=np.float64)
        for g in range(n_gp):
            theta_p = coords_p[g]
            acc_g = log_w_p[g]
            for s, members in enumerate(blocks):
                block_log = np.empty(qs, dtype=np.float64)
                for h, theta_s in enumerate(ts):
                    block_acc = log_ws[h]
                    for i in members:
                        row_mask = primary_map[i]
                        base = _item_base(
                            a_primary[i, row_mask],
                            float(a_specific[i]),
                            theta_p[row_mask],
                            float(theta_s),
                        )
                        block_acc += _cai2010_grm_category_logprobs(base, thresholds[i])[
                            int(y[p, i])
                        ]
                    block_log[h] = block_acc
                acc_g += _log_sum_exp(block_log)
            log_like_g[g] = acc_g
        loglik += _log_sum_exp(log_like_g)
    return loglik


def _cai2010_eq1_primary_map() -> np.ndarray:
    """Cai (2010) eq. (1), p. 586: eight-item pattern with cross-loadings."""
    return np.array(
        [
            [True, False],
            [True, False],
            [True, False],
            [True, True],
            [False, True],
            [False, True],
            [True, True],
            [False, True],
        ],
        dtype=bool,
    )


def _cai2010_eq1_specific_map() -> np.ndarray:
    """Doublets I1={1,2}, I2={3,4}, I3={5,6}, I4={7,8} (1-based item ids)."""
    return np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)


def _doublet_blocks(specific_map: np.ndarray) -> list[list[int]]:
    n_specific = int(np.max(specific_map) + 1)
    blocks: list[list[int]] = [[] for _ in range(n_specific)]
    for i, s in enumerate(specific_map):
        blocks[int(s)].append(int(i))
    return blocks


def _g2_fixture() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """p=1, S=2, six items, K=3, N=30; small quadrature."""
    rng = np.random.default_rng(2040)
    n_persons, n_items, n_cat = 30, 6, 3
    primary_map = np.ones((n_items, 1), dtype=bool)
    specific_map = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    a_primary = np.array([[1.2], [0.9], [1.0], [1.1], [0.8], [1.3]], dtype=np.float64)
    a_specific = np.array([0.7, 0.9, 0.8, 1.0, 0.6, 0.85], dtype=np.float64)
    thresholds = np.array(
        [
            [0.8, -0.6],
            [0.6, -0.8],
            [1.0, -0.4],
            [0.7, -0.7],
            [0.9, -0.5],
            [0.5, -1.0],
        ],
        dtype=np.float64,
    )
    phi = np.array([[1.0]], dtype=np.float64)
    y = rng.integers(0, n_cat, size=(n_persons, n_items), dtype=np.int64)
    for i in range(n_items):
        y[rng.integers(0, n_persons, size=3), i] = np.arange(n_cat)
    return y, a_primary, a_specific, thresholds, phi, primary_map, specific_map


def test_g1_cai2010_grm_link_matches_grm_logprobs() -> None:
    """G1: eq. (11)-(12), p. 589 vs ``poly::grm_logprobs`` at 1e-12."""
    try:
        from fast_mlsirm import _core
    except Exception:  # pragma: no cover
        pytest.skip("compiled core not available")
    if not hasattr(_core, "grm_cell_logprobs"):  # pragma: no cover
        pytest.skip("core built without grm cell")

    alpha = np.array([1.3, 0.1, -1.2])
    bases = np.array([-1.4, 0.0, 0.9, 2.2])
    a_p = np.array([1.1, -0.7])
    theta_p = np.array([0.35, -0.2])
    a_s = 0.85
    theta_s = -0.4
    latent = float(a_p @ theta_p + a_s * theta_s)
    for base in np.concatenate([bases, np.array([latent])]):
        paper = _cai2010_grm_category_logprobs(float(base), alpha)
        rust = np.asarray(_core.grm_cell_logprobs(float(base), alpha), dtype=np.float64)
        np.testing.assert_allclose(paper, rust, atol=1e-12)
        assert abs(float(np.log(np.exp(paper).sum()))) < 1e-12


def test_g1_reversed_alpha_is_non_finite() -> None:
    """Strictly increasing intercepts violate the ordered GRM boundary contract."""
    alpha = np.array([0.5, 1.2, 2.0])
    paper = _cai2010_grm_category_logprobs(0.0, alpha)
    assert not bool(np.all(np.isfinite(paper)))

    try:
        from fast_mlsirm import _core
    except Exception:  # pragma: no cover
        pytest.skip("compiled core not available")
    if not hasattr(_core, "grm_cell_logprobs"):  # pragma: no cover
        pytest.skip("core built without grm cell")
    rust = np.asarray(_core.grm_cell_logprobs(0.0, alpha), dtype=np.float64)
    assert not bool(np.all(np.isfinite(rust)))


def test_g2_eq15_unreduced_matches_reduced_on_identical_nodes() -> None:
    """G2: eq. (15), p. 589 — paper unreduced sum vs crate reduced marginal."""
    y, a_p, a_s, thr, phi, pmap, smap = _g2_fixture()
    q_primary, q_specific = 5, 5
    n_persons, n_items = y.shape
    n_primary = a_p.shape[1]
    n_specific = int(np.max(smap) + 1)
    n_cat = thr.shape[1] + 1
    full = _cai2010_marginal_loglik_unreduced(
        y, a_p, a_s, thr, phi, pmap, smap, q_primary, q_specific
    )
    reduced_numpy = _cai2010_marginal_loglik_reduced(
        y, a_p, a_s, thr, phi, pmap, smap, q_primary, q_specific
    )
    assert np.isfinite(full) and np.isfinite(reduced_numpy)
    numpy_gap = abs(full - reduced_numpy)
    assert numpy_gap <= 1e-9, (
        f"eq.(15) NumPy reorder gap {numpy_gap:.3e} "
        f"(full={full}, reduced_numpy={reduced_numpy})"
    )

    try:
        from fast_mlsirm import _core
    except Exception:  # pragma: no cover
        pytest.skip("compiled core not available")
    if not hasattr(_core, "two_tier_grm_marginal_loglik"):  # pragma: no cover
        pytest.skip("core built without two_tier_grm_marginal_loglik")

    observed = np.ones(n_persons * n_items, dtype=bool)
    rust_reduced = float(
        _core.two_tier_grm_marginal_loglik(
            a_p.reshape(-1),
            a_s.reshape(-1),
            thr.reshape(-1),
            phi.reshape(-1),
            y.astype(np.int64).reshape(-1),
            observed,
            pmap.reshape(-1),
            smap.reshape(-1),
            n_persons,
            n_items,
            n_primary,
            n_specific,
            n_cat,
            q_primary,
            q_specific,
        )
    )
    assert np.isfinite(rust_reduced)
    crate_gap = abs(full - rust_reduced)
    assert crate_gap <= 1e-9, (
        f"eq.(15) paper-vs-crate gap {crate_gap:.3e} "
        f"(full={full}, rust_reduced={rust_reduced})"
    )


def test_g3_eq1_pattern_accepts_and_restores_four_doublets() -> None:
    """G3: eq. (1), p. 586 — cross-loaded primaries and four item doublets."""
    primary_map = _cai2010_eq1_primary_map()
    specific_map = _cai2010_eq1_specific_map()
    blocks = _doublet_blocks(specific_map)
    assert blocks == [[0, 1], [2, 3], [4, 5], [6, 7]]

    n_items, n_primary, n_specific, n_cat = 8, 2, 4, 3
    rng = np.random.default_rng(2040)
    y = rng.integers(0, n_cat, size=(24, n_items), dtype=np.int64)
    for i in range(n_items):
        y[rng.integers(0, 24, size=3), i] = np.arange(n_cat)

    fit = fit_two_tier_grm(
        y,
        primary_map,
        specific_map,
        n_cat,
        n_primary,
        n_specific,
        q_primary=5,
        q_specific=5,
        max_iter=2,
        tol=1e-4,
        n_starts=1,
        seed=2040,
    )
    assert fit.a_primary.shape == (n_items, n_primary)
    assert fit.a_specific.shape == (n_items,)
    assert bool(np.all(np.isfinite(fit.a_primary)))
    assert bool(np.all(np.isfinite(fit.a_specific)))


def test_g3_rejects_two_specific_factors_on_one_item() -> None:
    """Eq. (1) permits at most one specific loading per item (p. 586)."""
    bad_specific = np.array(
        [
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int64,
    )
    counts = np.count_nonzero(bad_specific, axis=1)
    assert counts[0] == 2
    assert not bool(np.all(counts <= 1))

    with pytest.raises(ValueError, match="at most one specific"):
        encode_specific_map_from_columns(bad_specific)
