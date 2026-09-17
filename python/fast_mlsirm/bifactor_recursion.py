# Copyright (c) 2026 ContextualWisdomLab. All rights reserved.
# SPDX-License-Identifier: MIT

"""Two-stage Lord-Wingersky score-distribution recursion for bifactor models.

Computes the total-score distribution ``P(X = r | theta_0)`` by within-domain
recursion conditional on the general and specific factors (integrated over
the specific factor) followed by across-domain convolution, with exact
direct enumeration available as a small-model oracle.

References:
    - Lord, F. M., & Wingersky, M. S. (1984). Comparison of IRT true-score
      and equipercentile observed-score "equatings." *Applied Psychological
      Measurement, 8*(4), 453–461.
      https://doi.org/10.1177/014662168400800409
"""

from __future__ import annotations

import numpy as np


def _load_bifactor_core():
    """Return the compiled Rust bifactor core module, or None if unavailable."""
    try:
        from ._bifactor_core_loader import bifactor_core
        return bifactor_core()
    except Exception:
        try:
            from . import _bifactor_core
            return _bifactor_core
        except Exception:
            try:
                from . import _core
                return _core
            except Exception:
                return None


def bifactor_lord_wingersky(
    a_general: np.ndarray,
    a_specific: np.ndarray,
    thresholds: np.ndarray,
    item_domains: np.ndarray,
    n_cat: int,
    n_domains: int,
    theta_general: np.ndarray,
    theta_specific: np.ndarray,
    weights_specific: np.ndarray,
) -> np.ndarray:
    """Compute total score distribution P(X = r | theta_0) via two-stage Lord-Wingersky recursion.

    Within-domain recursion is performed conditional on (theta_0, theta_s) and integrated over
    theta_s, followed by across-domain convolution.

    Implementation basis: Lord, F. M., & Wingersky, M. S. (1984). Comparison
    of IRT true-score and equipercentile observed-score "equatings."
    *Applied Psychological Measurement, 8*(4), 453–461.
    https://doi.org/10.1177/014662168400800409
    """
    a_g = np.asarray(a_general, dtype=np.float64)
    a_s = np.asarray(a_specific, dtype=np.float64)
    thr = np.asarray(thresholds, dtype=np.float64).reshape(-1)
    domains = np.asarray(item_domains, dtype=np.int64)
    th_g = np.asarray(theta_general, dtype=np.float64)
    th_s = np.asarray(theta_specific, dtype=np.float64)
    w_s = np.asarray(weights_specific, dtype=np.float64)

    core = _load_bifactor_core()
    if core is None or not hasattr(core, "bifactor_lord_wingersky"):
        raise RuntimeError("bifactor Lord-Wingersky requires compiled Rust core")

    flat = np.asarray(
        core.bifactor_lord_wingersky(
            a_g, a_s, thr, domains, n_cat, n_domains, th_g, th_s, w_s
        ),
        dtype=np.float64,
    )
    total_max_score = len(a_g) * (n_cat - 1)
    return flat.reshape(len(th_g), total_max_score + 1)


def direct_enumeration_bifactor(
    a_general: np.ndarray,
    a_specific: np.ndarray,
    thresholds: np.ndarray,
    item_domains: np.ndarray,
    n_cat: int,
    n_domains: int,
    theta_general: np.ndarray,
    theta_specific: np.ndarray,
    weights_specific: np.ndarray,
) -> np.ndarray:
    """Compute exact direct enumeration ground truth for small item sets."""
    a_g = np.asarray(a_general, dtype=np.float64)
    a_s = np.asarray(a_specific, dtype=np.float64)
    thr = np.asarray(thresholds, dtype=np.float64).reshape(-1)
    domains = np.asarray(item_domains, dtype=np.int64)
    th_g = np.asarray(theta_general, dtype=np.float64)
    th_s = np.asarray(theta_specific, dtype=np.float64)
    w_s = np.asarray(weights_specific, dtype=np.float64)

    core = _load_bifactor_core()
    if core is None or not hasattr(core, "direct_enumeration_bifactor"):
        raise RuntimeError("direct enumeration requires compiled Rust core")

    flat = np.asarray(
        core.direct_enumeration_bifactor(
            a_g, a_s, thr, domains, n_cat, n_domains, th_g, th_s, w_s
        ),
        dtype=np.float64,
    )
    total_max_score = len(a_g) * (n_cat - 1)
    return flat.reshape(len(th_g), total_max_score + 1)
