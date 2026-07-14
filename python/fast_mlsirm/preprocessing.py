"""Response preprocessing utilities.

`irtree_expand` implements the mapping-matrix pseudo-item expansion of
Jeon & De Boeck (2016, "A generalized item response tree model for
psychological assessments", Behavior Research Methods): a categorical
response decomposes into conditional binary pseudo-items along a response
tree; nodes off the taken path are missing by design. Their Eq. 9 shows the
resulting model is an ordinary (multidimensional) binary IRT model on the
expanded matrix — so the expansion is pure preprocessing and the marginal
estimator applies unchanged (off-path cells reuse the NaN missingness
contract).
"""

from __future__ import annotations

import numpy as np


def irtree_expand(
    responses: np.ndarray,
    mapping: np.ndarray,
    node_dims: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand categorical responses into binary pseudo-items via a tree map.

    ``responses`` is persons x items with integer categories ``0..C-1`` (NaN =
    missing). ``mapping`` is the nodes x categories tree matrix ``T`` with
    entries 0/1/NaN: ``T[n, c]`` is the binary pseudo-response of node ``n``
    when category ``c`` was chosen, NaN when the node is off the path.
    Returns ``(expanded, factor_id)``: persons x (items * nodes) pseudo-binary
    matrix (NaN = off-path or missing) and its trait-dimension mapping —
    node ``n`` of every item loads on dimension ``node_dims[n]`` (default:
    dimension ``n``, one trait per tree node, the canonical IRTree structure).
    """
    y = np.asarray(responses, dtype=float)
    t = np.asarray(mapping, dtype=float)
    if t.ndim != 2:
        raise ValueError("mapping must be nodes x categories")
    n_nodes, n_cats = t.shape
    finite = t[np.isfinite(t)]
    if finite.size and not np.all((finite == 0.0) | (finite == 1.0)):
        raise ValueError("mapping entries must be 0, 1, or NaN")
    obs = np.isfinite(y)
    if obs.any():
        vals = y[obs]
        if np.any(vals < 0) or np.any(vals >= n_cats) or np.any(vals != np.round(vals)):
            raise ValueError(f"responses must be integer categories in 0..{n_cats - 1}")
    n_persons, n_items = y.shape
    expanded = np.full((n_persons, n_items * n_nodes), np.nan)
    cat_idx = np.where(obs, y, 0).astype(int)
    for n in range(n_nodes):
        node_vals = t[n, cat_idx]  # (P, I): 0/1/NaN by chosen category
        node_vals = np.where(obs, node_vals, np.nan)
        expanded[:, n * n_items : (n + 1) * n_items] = node_vals
    if node_dims is None:
        node_dims = np.arange(n_nodes)
    node_dims = np.asarray(node_dims, dtype=np.int64)
    if node_dims.shape != (n_nodes,):
        raise ValueError("node_dims must have one entry per tree node")
    factor_id = np.repeat(node_dims, n_items)
    return expanded, factor_id
