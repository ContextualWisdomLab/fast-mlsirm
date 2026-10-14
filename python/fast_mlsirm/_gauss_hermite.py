"""Probabilists' Gauss-Hermite rule for N(0, 1) quadrature at any node count.

``numpy.polynomial.hermite_e.hermegauss`` evaluates its weights from scaled
Hermite polynomial values; with NumPy 2.x that formula overflows from
q = 371, so the weights sum to 0 or NaN and normalization yields NaN (#2110).
Node count controls the precision of the marginal-likelihood integral and must
not be capped (#1929), so when the NumPy rule is not finite this module falls
back to the Golub-Welsch eigenvalue construction that the Rust core uses
(``crates/mlsirm-core/src/quadrature.rs::gauss_hermite_probabilists``).
Wherever the NumPy rule is finite it is returned unchanged, so results that
were already well defined do not move.

Method basis
------------
Golub and Welsch (1969, eqs. 2.1-2.2, pp. 222-223) show that the nodes of the
Gauss rule for a weight function whose monic orthogonal polynomials satisfy
``p_j(x) = (x - alpha_j) p_{j-1}(x) - beta_j p_{j-2}(x)`` are the eigenvalues
of the symmetric tridiagonal Jacobi matrix with diagonal ``alpha_j`` and
off-diagonal ``sqrt(beta_j)``, and (eq. 2.6, p. 223) that each weight is the
total mass of the weight function times the squared first component of the
corresponding normalized eigenvector. For the probabilists' Hermite weight
``exp(-x**2 / 2)`` the recurrence ``He_{k+1}(x) = x He_k(x) - k He_{k-1}(x)``
(Abramowitz & Stegun, 1972, Table 22.7, row 22.7.14, p. 782) gives
``alpha_k = 0`` and ``beta_k = k``. Weights are renormalized to sum to 1,
which removes the constant total mass ``sqrt(2 * pi)``.

References
----------
Abramowitz, M., & Stegun, I. A. (Eds.). (1972). *Handbook of mathematical
functions with formulas, graphs, and mathematical tables* (10th printing).
U.S. Government Printing Office.

Golub, G. H., & Welsch, J. H. (1969). Calculation of Gauss quadrature rules.
*Mathematics of Computation, 23*(106), 221-230.
https://doi.org/10.1090/S0025-5718-69-99647-1
"""

from __future__ import annotations

import numpy as np


def _golub_welsch_probabilists(q: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``q``-point rule from the Jacobi-matrix eigendecomposition."""
    if q == 1:
        return np.zeros(1), np.ones(1)
    off_diagonal = np.sqrt(np.arange(1, q, dtype=np.float64))
    jacobi = np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
    nodes, vectors = np.linalg.eigh(jacobi)
    weights = vectors[0] ** 2
    weights = weights / weights.sum()
    # The weight function is symmetric about 0, so the exact rule is too;
    # average mirrored pairs to remove eigensolver roundoff asymmetry, as the
    # Rust core does.
    nodes = (nodes - nodes[::-1]) / 2.0
    weights = (weights + weights[::-1]) / 2.0
    if q % 2 == 1:
        nodes[q // 2] = 0.0
    return nodes, weights


def probabilists_gauss_hermite(q: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``q`` ascending N(0, 1) quadrature nodes and unit-sum weights.

    Any ``q >= 1`` is accepted; the caller validates the node count.
    """
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        nodes, weights = np.polynomial.hermite_e.hermegauss(q)
        total = weights.sum()
    if np.isfinite(total) and total > 0.0 and np.isfinite(nodes).all():
        normalized = weights / total
        if np.isfinite(normalized).all():
            return nodes, normalized
    return _golub_welsch_probabilists(q)
