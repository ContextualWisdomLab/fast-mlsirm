# Adaptive exploratory factor rotation

`fast-mlsirm` treats factor rotation as two separate decisions:

1. **Find the best observed solution for one criterion.** Each criterion is
   optimized from deterministic identity and seeded random starts on the
   orthogonal or oblique manifold. The result reports projected-gradient
   stationarity, every start value, local-basin support, and the number of
   distinct observed minima.
2. **Choose among criteria with criterion-neutral evidence.** Objective values
   from varimax, geomin, oblimin, target, entropy, and other families have
   different scales and meanings, so the selector never compares them directly.
   It compares their solutions using simple-structure complexity, factor-size
   collapse, factor-correlation degeneracy, convergence, basin support,
   signed-permutation-aligned bootstrap congruence, and optional theory-target
   recovery.

There is no universally optimal rotation criterion. The selected criterion is
conditional on the extraction model, candidate set, population structure,
bootstrap design, and an explicit decision policy.

## Quick start

```python
import numpy as np
from fast_mlsirm import rotate_factor_loadings, select_rotation_criterion

unrotated = np.asarray(
    [
        [0.72, 0.39],
        [0.65, 0.35],
        [0.60, 0.31],
        [-0.31, 0.70],
        [-0.28, 0.64],
        [-0.25, 0.58],
    ]
)

solution = rotate_factor_loadings(
    unrotated,
    "geomin",
    mode="oblique",
    n_starts=64,
    seed=20260803,
)

selection = select_rotation_criterion(
    unrotated,
    ["quartimin", "geomin", "crawford_ferguson", "bentler"],
    mode="oblique",
    policy="fully_exploratory",
    n_starts=64,
)
print(selection.selected_criterion)
print(selection.evidence_grade)
print(selection.warning)
```

A single-sample selection is labelled `single_sample_diagnostic`. Provide a
`(replicates, variables, factors)` bootstrap-loading array to estimate
signed-permutation-aligned Tucker congruence. Fewer than 20 replicates are
labelled `bootstrap_exploratory`; 20 or more are
`bootstrap_supported`. These labels describe evidence quantity, not universal
truth.

## Built-in criterion registry

The first production slice covers the differentiable analytic catalogue needed
for extensible gradient projection:

| Family | Public criteria |
|---|---|
| Orthomax | `quartimax`, `varimax`, continuous `orthomax`, `varimin` |
| Crawford–Ferguson | continuous `crawford_ferguson`, `equamax`, `parsimax`, `factor_parsimony` |
| Direct oblimin | continuous `oblimin`, `quartimin`, `biquartimin`, `covarimin` |
| Geometric mean | `geomin` |
| Target | complete/NaN-partial `target`, weighted `pst` |
| Information | `entropy`, `infomax`, `mccammon` |
| Component loss | `simplimax`, `lp_wls` kernel |
| Bifactor | `bifactor`, `bigeomin` |
| Tandem | `tandem_i`, `tandem_ii` |
| Invariant simplicity | `oblimax`, `bentler` |

`available_rotation_criteria()` returns machine-readable capability metadata.
Continuous parameters such as Crawford–Ferguson `kappa`, Orthomax/Oblimin
`gamma`, and Geomin `delta` are part of the public API. A criterion can therefore
be added to the Rust registry without duplicating the optimizer.

The next catalogue expansion should add iterative/derivative-free wrappers such
as Promax power targets, Cubimax, full Lp/forced-simple-structure iteration,
cluster rotation, EIV/echelon analytic targets, and user-defined Rust/plugin
criteria. Those methods are not falsely advertised as implemented by this
slice.

## Optimization contract

For an orthogonal transform `T`, the pattern is

\[
\Lambda = A T, \qquad T^\top T = I.
\]

For an oblique unit-column transform `T`, the GPArotation convention is

\[
\Lambda = A T^{-\top}, \qquad \Phi=T^\top T,
\]

and the structure matrix is

\[
\Lambda_s = \Lambda\Phi.
\]

The Rust core uses analytic loading-space gradients, Barzilai–Borwein step
sizes, non-monotone Armijo line search, a Cayley orthogonal retraction, and unit-
column oblique projection. Starts are generated deterministically from a
SplitMix64/Box–Muller Gaussian stream and solved in coarse CPU threads. This
minimizes context switching while preserving exact reproducibility for a fixed
seed and worker-independent start order.

A finite start set cannot prove a global optimum. `basin_support` is the number
of starts within the requested relative objective tolerance of the best
observed value; `distinct_minima` counts observed objective basins. Increase
`n_starts`, inspect basin support, and bootstrap the extraction before relying
on a solution in high-stakes work.

## Criterion-neutral selection metrics

For pattern row \(i\), let \(s_i=\sum_j\lambda_{ij}^2\). The soft row-complexity
metric is

\[
C(\Lambda)=
\frac{\sum_i\left(s_i^2-\sum_j\lambda_{ij}^4\right)}
     {\sum_i s_i^2}.
\]

It is zero for exact one-loading-per-row structure and increases with
cross-loading energy. The selector also reports:

- smallest/largest factor sum-of-squares ratio (`factor_balance`),
- maximum absolute off-diagonal \(\Phi\),
- converged-start fraction,
- best-basin support fraction,
- mean and minimum-factor bootstrap Tucker congruence after sign/permutation
  alignment,
- optional target RMSE after alignment.

The selector converts each active metric to deterministic normalized ranks.
This prevents incompatible units from being mixed. Policies only determine the
rank weights:

- `interpretability_first`
- `stability_first`
- `theory_guided`
- `fully_exploratory`
- `recovery_first`
- `sparse_simple_structure`
- `bifactor_discovery`

Every candidate also receives a Pareto-frontier flag. The complete candidate
table is retained so a buyer can audit a decision or change the policy without
rerunning extraction.

## Bifactor caution

`bifactor` and `bigeomin` reserve the first loading column as a general factor;
canonicalization may sign-reflect it but never moves it. This is a rotational
convention, not evidence that a substantive general factor exists. Compare
bifactor, correlated-factor, second-order, testlet, and latent-space models with
held-out prediction, relation-safe model comparison, scoreability indices, and
parameter recovery before reporting a general score.

## GPU and CPU execution

This initial numerical path is Rust CPU with coarse multi-start parallelism.
The result names the backend explicitly as
`rust_cpu_coarse_multithreaded`; it does not silently claim GPU execution.
Batched criterion evaluation and multi-start optimization are natural GPU work,
but a GPU backend must first demonstrate objective/gradient parity, stationarity
parity, and identical selected basins on representative matrices. Until that
separate verified implementation lands, explicit backend provenance prevents
false acceleration claims.

## References (APA 7th)

Bernaards, C. A., & Jennrich, R. I. (2005). Gradient projection algorithms and
software for arbitrary rotation criteria in factor analysis. *Educational and
Psychological Measurement, 65*(5), 676–696.
https://doi.org/10.1177/0013164404272507

Browne, M. W. (2001). An overview of analytic rotation in exploratory factor
analysis. *Multivariate Behavioral Research, 36*(1), 111–150.
https://doi.org/10.1207/S15327906MBR3601_05

Crawford, C. B., & Ferguson, G. A. (1970). A general rotation criterion and its
use in orthogonal rotation. *Psychometrika, 35*(3), 321–332.
https://doi.org/10.1007/BF02310792

Jennrich, R. I., & Bentler, P. M. (2011). Exploratory bi-factor analysis.
*Psychometrika, 76*(4), 537–549. https://doi.org/10.1007/s11336-011-9218-4

Kiers, H. A. L. (1994). Simplimax: Oblique rotation to an optimal target with
simple structure. *Psychometrika, 59*(4), 567–579.
https://doi.org/10.1007/BF02294392

Myers, N. D., Ahn, S., & Jin, Y. (2013). Rotation to a partially specified
target matrix in exploratory factor analysis: How many targets?
*Structural Equation Modeling, 20*(1), 131–147.
https://doi.org/10.1080/10705511.2013.742399

Sass, D. A., & Schmitt, T. A. (2010). A comparative investigation of rotation
criteria within exploratory factor analysis. *Multivariate Behavioral Research,
45*(1), 73–103. https://doi.org/10.1080/00273170903504810

Weide, A. C., & Beauducel, A. (2019). Varimax rotation based on gradient
projection is a feasible alternative to SPSS. *Frontiers in Psychology, 10*,
645. https://doi.org/10.3389/fpsyg.2019.00645
