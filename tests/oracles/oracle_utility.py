"""Independent scipy oracle for selection utility analysis.

Regenerates every numeric fixture pinned in ``tests/unit/utility_tests.rs``
and ``tests/test_paper_features.py::TestSelectionUtility``. Run with any
Python that has numpy + scipy (values were pinned with scipy 1.x)::

    python tests/oracles/oracle_utility.py

Quantities:

- ``ux(sr) = phi(xc)/sr``, ``xc = Phi^-1(1-sr)`` (truncated-normal
  selection intensity),
- ``pux = rxy * ux(sr)`` (Naylor-Shine),
- BCG utility ``n*period*sdy*pux - cost_total``,
- Taylor-Russell success ratio ``Q(xc, yc, rho)/sr``, ``yc = Phi^-1(1-br)``.

``Q`` is computed two independent ways:

1. the multivariate-normal CDF identity
   ``Q(h,k,rho) = 1 - Phi(h) - Phi(k) + F(h,k)``, and
2. adaptive quadrature (``scipy.integrate.quad``, epsrel 1e-12) on the same
   conditional-normal integral the Rust core uses,
   ``int_h^inf phi(x) Phi((rho*x - k)/sqrt(1-rho^2)) dx`` — this is the
   oracle for the near-degenerate-|rho| regression fixtures.
"""

import math

from scipy.integrate import quad
from scipy.stats import multivariate_normal, norm


def q_mvn(h, k, rho):
    f = multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]]).cdf([h, k])
    return 1.0 - norm.cdf(h) - norm.cdf(k) + f


def q_quad(h, k, rho):
    s = math.sqrt(1.0 - rho * rho)
    val, _ = quad(
        lambda x: norm.pdf(x) * norm.cdf((rho * x - k) / s),
        h,
        math.inf,
        epsabs=1e-18,
        epsrel=1e-12,
        limit=1000,
    )
    return val


def ux(sr):
    xc = norm.ppf(1 - sr)
    return norm.pdf(xc) / sr


def taylor_russell(rxy, sr, br, q=q_mvn):
    xc = norm.ppf(1 - sr)
    yc = norm.ppf(1 - br)
    qv = q(xc, yc, rxy)
    return qv / sr, qv


def bcg(n, sdy, rxy, sr, cost_total, period):
    return n * period * sdy * rxy * ux(sr) - cost_total


if __name__ == "__main__":
    print("=== ux / pux fixtures ===")
    for sr in [0.05, 0.3, 0.5, 0.9]:
        print(f"sr={sr}: xc={norm.ppf(1 - sr):.15f} ux={ux(sr):.15f}")
    print(f"pux(rxy=.5, sr=.3) = {0.5 * ux(0.3):.15f}")

    print("\n=== Taylor-Russell fixtures (mvn-CDF oracle) ===")
    for (rxy, sr, br) in [
        (0.5, 0.5, 0.6),
        (0.5, 0.3, 0.2),
        (0.3, 0.05, 0.8),
        (-0.6, 0.3, 0.5),
        (0.8, 0.3, 0.2),
        (0.0, 0.3, 0.7),
    ]:
        s, q = taylor_russell(rxy, sr, br)
        print(f"rxy={rxy} sr={sr} br={br}: success={s:.15f} q={q:.15f}")

    print("\n=== BCG fixtures ===")
    print(f"bcg(1, 10000, .5, .3, 0, 1)   = {bcg(1, 10000, .5, .3, 0, 1):.10f}")
    print(f"bcg(50, 8000, .4, .2, 25000, 3) = {bcg(50, 8000, .4, .2, 25000, 3):.10f}")

    print("\n=== near-degenerate-rho regression fixtures (quad oracle) ===")
    for (rxy, sr, br) in [(-0.999999, 0.9, 0.9), (0.999999, 1e-12, 1e-12)]:
        s, q = taylor_russell(rxy, sr, br, q=q_quad)
        print(f"rxy={rxy} sr={sr} br={br}: success={s:.15f} q={q:.20g}")

    print("\n=== mutation predictions ===")
    print(f"M1 (drop rxy) pux at sr=.3: {ux(0.3):.15f} vs good {0.5 * ux(0.3):.15f}")
    xc03, yc08 = norm.ppf(1 - 0.3), norm.ppf(1 - 0.8)
    print(
        "M4 (sr<->br role swap) at rho=.5, sr=.3, br=.8: "
        f"good success={q_mvn(xc03, yc08, 0.5) / 0.3:.15f} "
        f"mutant={q_mvn(yc08, xc03, 0.5) / 0.8:.15f}"
    )
