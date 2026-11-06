"""Monte Carlo check for joint multigroup Oakes SEs (run on s1).

Usage: python scripts/bifactor_multigroup_oakes_calibration.py --replicates R
    --persons-per-group N --q-general QG --q-specific QS --fd-step H

The bootstrap interval measures Monte Carlo uncertainty in SD/mean-SE across
the R independent simulated data sets. Oakes (1999, eq. 6, p. 480);
Cai et al. (2011, p. 230); Gibbons et al. (2007, eqs. 9, 15, pp. 7, 9).
References (APA 7th): Oakes, D. (1999). Direct calculation of the information
matrix via the EM algorithm. *Journal of the Royal Statistical Society: Series
B, 61*(2), 479–482. https://doi.org/10.1111/1467-9868.00188
Cai, L., Yang, J. S., & Hansen, M. (2011). Generalized full-information item
bifactor analysis. *Psychological Methods, 16*(3), 221–248.
https://doi.org/10.1037/a0023350
Gibbons, R. D., Bock, R. D., Hedeker, D., Weiss, D. J., Segawa, E.,
Bhaumik, D. K., Kupfer, D. J., Frank, E., Grochocinski, V. J., & Stover,
A. (2007). Full-information item bifactor analysis of graded response data.
*Applied Psychological Measurement, 31*(1), 4–19.
https://doi.org/10.1177/0146621606289485
"""

import argparse
from types import SimpleNamespace
import numpy as np
from fast_mlsirm import fit_bifactor_grm_multigroup, bifactor_multigroup_oakes_se


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, required=True)
    parser.add_argument("--persons-per-group", type=int, required=True)
    parser.add_argument("--q-general", type=int, required=True)
    parser.add_argument("--q-specific", type=int, required=True)
    parser.add_argument("--fd-step", type=float, required=True)
    parser.add_argument("--compare-q-general", type=int)
    parser.add_argument("--compare-q-specific", type=int)
    parser.add_argument("--precision-only", action="store_true")
    parser.add_argument("--seed", type=int, default=2113)
    args = parser.parse_args()
    if args.replicates < 3 or args.persons_per_group < 1:
        parser.error("at least 3 replicates and 1 person per group are required")
    if args.precision_only and (not args.compare_q_general or not args.compare_q_specific):
        parser.error("--precision-only requires both --compare-q controls")
    rng = np.random.default_rng(args.seed)
    smap = np.array([-1] + [0] * 4 + [1] * 4 + [2] * 4)
    anchor = np.ones(13, dtype=bool)
    ag = np.linspace(0.8, 1.2, 13)
    as_ = np.where(smap < 0, 0.0, 0.7)
    d = np.tile(np.array([1.0, 0.0, -1.0]), (13, 1))
    estimate, reported_se = [], []
    for rep in range(args.replicates):
        group = np.repeat(np.arange(3), args.persons_per_group)
        theta_g = rng.normal(np.array([0.0, 0.25, -0.2])[group],
                             np.array([1.0, 1.1, 0.9])[group])
        theta_s = rng.standard_normal((group.size, 3))
        y = np.empty((group.size, 13), dtype=np.int64)
        for i, s in enumerate(smap):
            base = ag[i] * theta_g + (0 if s < 0 else as_[i] * theta_s[:, s])
            tail = 1 / (1 + np.exp(-(base[:, None] + d[i])))
            prob = np.concatenate((1-tail[:, :1], tail[:, :-1]-tail[:, 1:],
                                   tail[:, -1:]), axis=1)
            cdf = np.cumsum(prob, axis=1)
            y[:, i] = np.sum(rng.random(group.size)[:, None] > cdf, axis=1)
        if args.precision_only:
            fit = SimpleNamespace(
                a_general=np.tile(ag, (3, 1)), a_specific=np.tile(as_, (3, 1)),
                threshold=np.tile(d, (3, 1, 1)),
                general_mean=np.array([0.0, 0.25, -0.2]),
                general_sd=np.array([1.0, 1.1, 0.9]),
                specific_sd=np.ones((3, 3)), n_groups=3, n_specific=3, n_cat=4,
            )
            base = bifactor_multigroup_oakes_se(
                fit, y, group, smap, anchor, q_general=args.q_general,
                q_specific=args.q_specific, fd_step=args.fd_step)
            print(f"q={args.q_general},{args.q_specific} PD={base.positive_definite} "
                  f"reason={base.non_pd_reason}", flush=True)
            if not base.positive_definite:
                raise RuntimeError(base.non_pd_reason)
            refined = bifactor_multigroup_oakes_se(
                fit, y, group, smap, anchor,
                q_general=args.compare_q_general,
                q_specific=args.compare_q_specific, fd_step=args.fd_step)
            print(f"q={args.compare_q_general},{args.compare_q_specific} "
                  f"PD={refined.positive_definite} reason={refined.non_pd_reason}", flush=True)
            if not refined.positive_definite:
                raise RuntimeError(refined.non_pd_reason)
            drift = np.max(np.abs(refined.se-base.se) / refined.se)
            print(f"max_relative_SE_change={drift:.6g} "
                  f"tolerance=0.05 pass={drift < 0.05}")
            return
        fit = fit_bifactor_grm_multigroup(
            y, group, smap, 4, 3, anchor, q_general=args.q_general,
            q_specific=args.q_specific, max_iter=500, tol=1e-6,
            n_starts=1, seed=args.seed+rep)
        if not fit.converged:
            raise RuntimeError(f"replicate {rep} did not converge")
        oakes = bifactor_multigroup_oakes_se(
            fit, y, group, smap, anchor, q_general=args.q_general,
            q_specific=args.q_specific, fd_step=args.fd_step)
        if not oakes.positive_definite:
            raise RuntimeError(f"replicate {rep}: {oakes.non_pd_reason}")
        if rep == 0 and args.compare_q_general and args.compare_q_specific:
            refined = bifactor_multigroup_oakes_se(
                fit, y, group, smap, anchor,
                q_general=args.compare_q_general,
                q_specific=args.compare_q_specific, fd_step=args.fd_step)
            if not refined.positive_definite:
                raise RuntimeError(f"refined quadrature: {refined.non_pd_reason}")
            drift = np.max(np.abs(refined.se-oakes.se) / refined.se)
            print(f"q_stability_max_relative_SE_change={drift:.6g} "
                  f"tolerance=0.05 pass={drift < 0.05}", flush=True)
        j = oakes.labels.index("a_general:0")
        estimate.append(fit.a_general[0, 0])
        reported_se.append(oakes.se[j])
        print(f"replicate={rep+1} estimate={estimate[-1]:.6g} se={reported_se[-1]:.6g}",
              flush=True)
    estimate = np.asarray(estimate)
    reported_se = np.asarray(reported_se)
    ratio = estimate.std(ddof=1) / reported_se.mean()
    # Paired resampling preserves the estimate/SE association.
    draw = rng.integers(0, args.replicates, size=(2000, args.replicates))
    ratios = estimate[draw].std(axis=1, ddof=1) / reported_se[draw].mean(axis=1)
    lo, hi = np.quantile(ratios, [0.025, 0.975])
    print(f"R={args.replicates} empirical_SD/mean_Oakes_SE={ratio:.6g} "
          f"MC_95pct_interval=[{lo:.6g}, {hi:.6g}]")


if __name__ == "__main__":
    main()
