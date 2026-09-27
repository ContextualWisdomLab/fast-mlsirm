"""Continuous known-population recovery for the compiled six-latent focal fit.

Cai (2010), DOI 10.1007/s11336-010-9178-0, p.587 eqs.4-7,
p.588 eq.9, p.589 eqs.11-12: independent Gaussian latent factors and
conditionally independent categorical GRM responses. Appendix A, pp.608-609,
supplies the fitted Gaussian objective. This independent test-only generator
uses physical continuous latent draws, not the fitting quadrature grid.

Opened Python 3.12 manual, random.gauss, random.choices and Notes on
Reproducibility: https://docs.python.org/3.12/library/random.html.
Opened exact CPython 3.12.11 implementation, Random.gauss/choices:
https://github.com/python/cpython/blob/v3.12.11/Lib/random.py.
Opened Python 3.12 itertools.pairwise for consecutive category differences:
https://docs.python.org/3.12/library/itertools.html#itertools.pairwise.
Use one private Random instance in one thread; record Python version and input
hash because gauss/choices are not guaranteed unchanged across Python versions.

The bank, seed, sample size, grids, cap, convergence tolerance and recovery
bounds are prespecified test choices, not published scientific cutoffs. One
synthetic population realization is not coverage, real-data identification,
actual-study convergence, or acceptance of a library release.
"""

import hashlib
import itertools
import json
import math
import random
import sys

import numpy as np
from fast_mlsirm import fit_two_tier_grm_focal_orthogonal


def test_native_six_latent_continuous_gaussian_recovery(device="cpu"):
    """Recover all six nonstandard Gaussian means/SDs at two node counts.

    Sources and implementation-choice limits are in the module docstring.
    GRM probabilities below independently instantiate Cai p.589 eqs.11-12
    only to generate synthetic test responses; estimation remains native Rust.
    """
    seed = 20260927
    rng = random.Random(seed)
    mean = np.array([0.30, -0.25, 0.20, -0.30, 0.35, -0.15])
    sd = np.array([1.15, 0.85, 0.80, 1.20, 1.10, 0.90])
    wording = {4, 5, 6, 9, 12, 14, 15}
    ap = np.array(
        [
            [a, 0.9 if i in wording else 0.0]
            for i, a in enumerate([0.9, 1.3, 0.7, 1.1] * 4)
        ]
    )
    asp = np.array([1.1, 0.8, 1.4, 1.0] * 4)
    sm = np.repeat(np.arange(4), 4)
    threshold = np.tile([1.2, 0.0, -1.2], (16, 1))
    rows = []
    for _ in range(4096):
        theta = [rng.gauss(float(m), float(s)) for m, s in zip(mean, sd)]
        row = []
        for i in range(16):
            eta = ap[i, 0] * theta[0] + ap[i, 1] * theta[1] + asp[i] * theta[2 + sm[i]]
            cumulative = (
                [1.0]
                + [1.0 / (1.0 + math.exp(-(eta + d))) for d in threshold[i]]
                + [0.0]
            )
            probability = [a - b for a, b in itertools.pairwise(cumulative)]
            assert min(probability) >= 0.0
            assert abs(sum(probability) - 1.0) < 1e-14
            row.append(rng.choices(range(4), weights=probability, k=1)[0])
        rows.append(row)
    responses = np.array(rows, dtype=np.int64)
    digest = hashlib.sha256(responses.astype("<i8").tobytes()).hexdigest()
    fits = []
    for nodes in (15, 21):
        fit = fit_two_tier_grm_focal_orthogonal(
            responses,
            ap != 0,
            sm,
            ap,
            asp,
            threshold,
            np.zeros(6),
            np.ones(6),
            n_cat=4,
            n_primary=2,
            n_specific=4,
            q_primary=nodes,
            q_specific=nodes,
            max_iter=500,
            tol=1e-6,
            device=device,
        )
        print(
            json.dumps(
                {
                    "python": sys.version,
                    "seed": seed,
                    "input_sha256": digest,
                    "nodes": nodes,
                    "mean": fit.latent_mean.tolist(),
                    "sd": fit.latent_sd.tolist(),
                    "updates": fit.n_iter,
                    "reason": fit.termination_reason,
                    "final_loglik_change": fit.final_loglik_change,
                }
            ),
            flush=True,
        )
        assert fit.converged, fit.termination_reason
        assert fit.termination_reason == "tolerance_met"
        assert np.max(np.abs(fit.latent_mean - mean)) < 0.25
        assert np.max(np.abs(fit.latent_sd - sd)) < 0.25
        fits.append(fit)
    assert np.max(np.abs(fits[0].latent_mean - fits[1].latent_mean)) < 0.04
    assert np.max(np.abs(fits[0].latent_sd - fits[1].latent_sd)) < 0.04
