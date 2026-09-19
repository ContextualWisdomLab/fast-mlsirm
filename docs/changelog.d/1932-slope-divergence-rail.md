# Source-backed slope-divergence guard replaces the unsourced 10.0 ceiling (#1932)

## Fixed

- **The unsourced `10.0` slope-magnitude bound is gone.** Every slope M-step
  in the crate (`crate::mmle`, `crate::twopl`, `crate::mhrm`,
  `crate::testlet`, `crate::mixture`, `crate::mixed`, plus the NumPy MMLE
  reference in `python/fast_mlsirm/estimators/mmle.py`) clamped to the same
  `10.0` constant that no source states as a ceiling on discrimination
  parameters. The clamp is now a single shared numerical safety rail,
  `SLOPE_DIVERGENCE_RAIL = 30.0` (canonical home: `crate::mmle`; the other
  modules alias it, so one item no longer has a different reachable range
  depending on which entry point fitted it). The value carries no measurement
  meaning: once the complete linear predictor reaches
  `|a * theta + b| >= 36`, the logistic is saturated to machine precision.
  The unrestricted intercept means a rail slope alone does not imply
  saturation at every latent node.
- **Divergence is reported, never passed off as an estimate.** A slope resting
  on the rail with its M-step still pushing outward — a Heywood-like boundary
  solution — now forces non-convergence with a reason instead of clamping
  silently: `converged = false` with `termination_reason = "slope_diverged"`,
  plus a per-item (per-(class, item) for mixtures) `slope_diverged` flag on
  every affected result (`MmleResult`, `TwoPlResult`, `MhrmResult`,
  `TestletResult`, `MixtureResult`; `MixedFit` keeps reporting through
  `MixedItemEstimate::at_bound` and now also refuses to claim convergence).
  The new `slope_diverged` key is exposed on the `fit_2pl`, `fit_mhrm`,
  `fit_mixture`, and `fit_testlet` Python dicts (the `fit_mmle_2pl` tuple
  keeps its shape; its `converged` entry already flips to `False`). The rail
  stays symmetric everywhere, so reverse-keyed (negative) slopes keep their
  sign.
- **No behavior change for well-conditioned fits.** The clamp is a no-op
  unless an M-step proposal would leave the rail, so interior trajectories —
  including every standard recovery fixture — are untouched. Only fits that
  used to pin silently at `10.0` now travel to the rail and report divergence.
  Identical response columns alone are not treated as proof of infinite
  discrimination: although they violate local independence, each column can
  still have an interior marginal slope. MH-RM reports divergence only when an
  actual update exceeds the rail, including symmetrically for negative slopes.
- **References.** Bock, R. D., & Aitkin, M. (1981). Marginal maximum
  likelihood estimation of item parameters: Application of an EM algorithm.
  *Psychometrika, 46*(4), 443–459. https://doi.org/10.1007/BF02293801 —
  boundary solutions with infinite slopes are Heywood cases (p. 457); ML
  gives non-finite values for degenerate patterns and Newton can fail there
  (p. 454). Mislevy, R. J. (1985). *Bayes modal estimation in item response
  models* (ETS Research Report No. 85–33). Educational Testing Service.
  https://eric.ed.gov/?id=ED268138 — ML yields infinite or implausible
  estimates in small samples (p. 44); priors pull extreme ill-determined
  values toward the center (p. 39); published as Mislevy (1986),
  *Psychometrika, 51*(2), 177–195. Chalmers, R. P. (2012). mirt: A
  multidimensional item response theory package for the R environment.
  *Journal of Statistical Software, 48*(6), 1–29.
  https://doi.org/10.18637/jss.v048.i06 — prior constraints (MAP) for
  excessive, convergence-problem parameters (pp. 14–15).
