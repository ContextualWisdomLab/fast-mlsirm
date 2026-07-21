"""Response preprocessing utilities.

`irtree_expand` implements the mapping-matrix pseudo-item expansion described
by Jeon and De Boeck (2016): a categorical response decomposes into conditional
binary internal outcomes, and nodes off the unique path are missing by design.
Their generalized model permits richer node-specific structures than this
repository's estimator; this helper only performs the mapping and does not
claim to implement every model in the article.

Reference (APA 7th ed.):
    Jeon, M., & De Boeck, P. (2016). A generalized item response tree model
        for psychological assessments. *Behavior Research Methods, 48*(3),
        1070–1085. https://doi.org/10.3758/s13428-015-0631-y
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
    if y.ndim != 2:
        raise ValueError("responses must be a persons x items matrix")
    n_persons, n_items = y.shape
    if n_persons == 0 or n_items == 0:
        raise ValueError("responses must contain at least one person and one item")
    if np.any(np.isinf(y)):
        raise ValueError("responses must contain integer categories or NaN")

    t = np.asarray(mapping, dtype=float)
    if t.ndim != 2:
        raise ValueError("mapping must be nodes x categories")
    n_nodes, n_cats = t.shape
    if n_nodes == 0 or n_cats == 0:
        raise ValueError("mapping must contain at least one node and one category")
    if np.any(~(np.isnan(t) | (t == 0.0) | (t == 1.0))):
        raise ValueError("mapping entries must be 0, 1, or NaN")

    # Bound the dense expansion so untrusted item/node counts cannot force a
    # multi-GB allocation (Jeon-De Boeck expansion is (persons, items*nodes)).
    # Byte budget (not a raw element count): the dense float64 output plus
    # per-node temporaries dominate memory; 64 MiB covers realistic IRTrees
    # (31k x 57 x a few nodes) while blocking allocation-DoS inputs.
    MAX_EXPANDED_BYTES = 64 * 1024 * 1024
    if n_persons * n_items * n_nodes * 8 > MAX_EXPANDED_BYTES:
        raise ValueError(
            f"expanded matrix ({n_persons} x {n_items * n_nodes}) exceeds the "
            f"{MAX_EXPANDED_BYTES}-byte limit"
        )
    if np.any(np.all(np.isnan(t), axis=0)):
        raise ValueError("every response category must have a non-empty tree path")
    encoded_paths = np.where(np.isnan(t), -1.0, t).T
    if np.unique(encoded_paths, axis=0).shape[0] != n_cats:
        raise ValueError("response categories must have distinct tree paths")
    for node in t:
        node_values = node[np.isfinite(node)]
        if not (np.any(node_values == 0.0) and np.any(node_values == 1.0)):
            raise ValueError("every tree node must contain both binary branches")

    obs = ~np.isnan(y)
    if obs.any():
        vals = y[obs]
        if np.any(vals < 0) or np.any(vals >= n_cats) or np.any(vals != np.round(vals)):
            raise ValueError(f"responses must be integer categories in 0..{n_cats - 1}")
    expanded = np.full((n_persons, n_items * n_nodes), np.nan)
    cat_idx = np.where(obs, y, 0).astype(int)
    for n in range(n_nodes):
        node_vals = t[n, cat_idx]  # (P, I): 0/1/NaN by chosen category
        node_vals = np.where(obs, node_vals, np.nan)
        expanded[:, n * n_items : (n + 1) * n_items] = node_vals
    if node_dims is None:
        node_dims = np.arange(n_nodes)
    node_dims_arr = np.asarray(node_dims)
    if node_dims_arr.shape != (n_nodes,):
        raise ValueError("node_dims must have one entry per tree node")
    try:
        nd = node_dims_arr.astype(np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("node_dims must be finite non-negative integers") from exc
    if (
        not np.all(np.isfinite(nd))
        or np.any(nd < 0)
        or np.any(nd != np.floor(nd))
        or np.any(nd >= n_nodes)
    ):
        raise ValueError(
            "node_dims must be integer dimension indices in 0..number of nodes-1"
        )
    node_dims = nd.astype(np.int64)
    unique_dims = np.unique(node_dims)
    if not np.array_equal(unique_dims, np.arange(unique_dims.size)):
        raise ValueError("node_dims must use contiguous dimension indices starting at 0")
    factor_id = np.repeat(node_dims, n_items)
    return expanded, factor_id
