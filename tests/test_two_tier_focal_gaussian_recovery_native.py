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
from fast_mlsirm import (
    expected_total_score_two_tier_from_fit,
    fit_two_tier_grm,
    fit_two_tier_grm_focal_orthogonal,
    score_two_tier_grm_orthogonal,
)


def _continuous_six_latent_fixture(n_persons, mean, sd, seed):
    """Reuse Cai (2010) eqs.4–7,11–12 generator; see module sources.

    Test-only continuous draws; parameters and seed are explicit fixture inputs.
    """
    rng = random.Random(seed)
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
    for _ in range(n_persons):
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
    return responses, ap, asp, sm, threshold, digest


def test_native_six_latent_continuous_gaussian_recovery(
    device="cpu", gpu_memory_budget_bytes=None
):
    """Recover all six nonstandard Gaussian means/SDs at two node counts.

    Sources and implementation-choice limits are in the module docstring.
    GRM probabilities below independently instantiate Cai p.589 eqs.11-12
    only to generate synthetic test responses; estimation remains native Rust.
    """
    seed = 20260927
    mean = np.array([0.30, -0.25, 0.20, -0.30, 0.35, -0.15])
    sd = np.array([1.15, 0.85, 0.80, 1.20, 1.10, 0.90])
    responses, ap, asp, sm, threshold, digest = _continuous_six_latent_fixture(
        4096, mean, sd, seed
    )
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
            gpu_memory_budget_bytes=gpu_memory_budget_bytes,
            cache_item_tables=True,
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


def test_native_reference_focal_expected_score_pipeline(device="cpu"):
    """Fit reference items, fix that bank, score under the final focal prior.

    Cai (2010), pp.587–590 eqs.4–12 and pp.608–609 Appendices A/B supply
    the fitted model and posterior moments. Expected scores use the common
    reference density (STAT414 lesson26.1, cited/read in that API).
    Sample sizes, seeds, seven-node fit/score grid and 121/241 nuisance grids
    are synthetic bridge choices, not study precision or population recovery.
    All estimation and numerical scores are existing library calls.
    """
    reference_y, ap, _, sm, _, ref_hash = _continuous_six_latent_fixture(
        512, np.zeros(6), np.ones(6), 20260928,
    )
    focal_y, _, _, _, _, focal_hash = _continuous_six_latent_fixture(
        512, np.array([0.30, -0.25, 0.20, -0.30, 0.35, -0.15]),
        np.array([1.15, 0.85, 0.80, 1.20, 1.10, 0.90]), 20260929,
    )
    print(json.dumps({"phase": "inputs", "reference_sha256": ref_hash,
                      "focal_sha256": focal_hash, "device": device}), flush=True)
    reference = fit_two_tier_grm(
        reference_y, ap != 0, sm, n_cat=4, n_primary=2, n_specific=4,
        q_primary=7, q_specific=7, max_iter=2000, tol=1e-6,
        n_starts=2, seed=20260928, primary_correlation="identity",
    )
    print(json.dumps({"phase": "reference", "converged": reference.converged,
                      "reason": reference.termination_reason,
                      "updates": reference.n_iter}), flush=True)
    assert reference.converged, reference.termination_reason
    assert reference.primary_identification == "orthogonal"
    snapshots = [a.copy() for a in
                 (reference.a_primary, reference.a_specific, reference.threshold)]
    controls = dict(
        responses=focal_y, primary_map=ap != 0, specific_map=sm,
        a_primary=reference.a_primary, a_specific=reference.a_specific,
        threshold=reference.threshold, n_cat=4, n_primary=2, n_specific=4,
        q_primary=7, q_specific=7, device=device, cache_item_tables=True,
        gpu_memory_budget_bytes=(1 << 30) if device == "gpu" else None,
    )
    focal = fit_two_tier_grm_focal_orthogonal(
        **controls, latent_mean=np.zeros(6), latent_sd=np.ones(6),
        max_iter=2000, tol=1e-6,
    )
    print(json.dumps({"phase": "focal", "converged": focal.converged,
                      "reason": focal.termination_reason, "updates": focal.n_iter,
                      "backend": focal.scores.backend}), flush=True)
    assert focal.converged, focal.termination_reason
    assert focal.scores.backend == device
    scored = score_two_tier_grm_orthogonal(
        **controls, latent_mean=focal.latent_mean, latent_sd=focal.latent_sd,
    )
    np.testing.assert_allclose(scored.mean, focal.scores.mean, atol=1e-10, rtol=0)
    for actual, snapshot in zip(
        (reference.a_primary, reference.a_specific, reference.threshold), snapshots,
    ):
        np.testing.assert_array_equal(actual, snapshot)
    curves = [expected_total_score_two_tier_from_fit(
        reference, sm, scored.mean[:, 0], focal_primary=0, q_nuisance=q,
        orthogonal_primary_identification=True,
        primary_ref_mean=np.zeros(2), primary_ref_sd=np.ones(2),
        specific_ref_mean=np.zeros(4), specific_ref_sd=np.ones(4),
    ) for q in (121, 241)]
    np.testing.assert_array_equal(curves[0].theta_focal, scored.mean[:, 0])
    np.testing.assert_allclose(curves[0].expected_total, curves[1].expected_total,
                               atol=1e-8, rtol=0)
