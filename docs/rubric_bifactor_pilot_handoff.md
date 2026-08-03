# Generated-item pilot handoff to binary bifactor calibration

`fast_mlsirm.rubric.build_bifactor_pilot_design` converts replay-verified
pilot observations into a deterministic, content-addressed handoff for the
existing Rust-backed `BIFAC2PLM` estimator.

The handoff is deliberately narrow. It records a confirmatory loading pattern
and produces validated `fast_mlsirm.fit` arguments; it does not claim that the
pattern is identified, fits adequately, yields scoreable general or domain
scores, is fair, or is valid for a high-stakes use.

## Loading contract

The design preserves the classical item-bifactor restriction:

- every item is declared to load on one shared general factor;
- every item is assigned to at most one specific factor; and
- the specific factor is the item's governed `query_testlet_id`.

The general factor has a descriptive nonnumeric identifier, defaulting to
`general_factor`. Specific factors retain their descriptive testlet identifiers
in sorted order. The zero-based integer vector passed to the numerical API is a
transport representation only; the descriptive mapping and complete pilot
provenance remain in the fingerprinted design artifact.

This follows the structural boundary described by Gibbons and Hedeker (1992):
every item has a loading on the primary dimension and at most one group factor.
The handoff does not infer the factor structure from response data.

## Example

```python
from fast_mlsirm import fit
from fast_mlsirm.rubric import build_bifactor_pilot_design

# records are PilotObservationRecord objects issued from admitted pilot items.
design = build_bifactor_pilot_design(
    records,
    general_factor_id="reasoning_general_factor",
)

result = fit(**design.to_fit_kwargs())
```

`to_fit_kwargs()` supplies:

- a fresh `float64` persons-by-items response matrix;
- `NaN` for missing, not-applicable, and insufficient-evidence cells;
- a fresh per-item specific-factor assignment vector; and
- `FitConfig(model="BIFAC2PLM", estimator="mmle", latent_dim=1)`.

A caller may provide a `FitConfig` to tune numerical controls, but the method
fails closed unless the model remains `BIFAC2PLM`, the estimator remains
marginal maximum likelihood, and the general-factor interaction remains
one-dimensional.

## Provenance and missingness

The builder delegates all response assembly to `build_mirt_pilot_design`, so it
inherits the same fail-closed contracts:

- one pilot study per design;
- one immutable provenance binding per item;
- at most one response per respondent-item cell;
- binary observed categories only;
- observed support for every retained respondent and item; and
- a bounded dense persons-by-items allocation.

Multi-rater cells must use `build_facets_pilot_design`; polytomous responses
must not be silently dichotomized. Exact response states and rater assignments
remain in the nested binary design even though all non-observed states are
represented numerically as `NaN` for estimation.

The outer bifactor fingerprint binds the general-factor identity, every item
assigned to it, the descriptive specific-factor mapping, and the complete
content-addressed binary design.

## Required downstream gates

A successful handoff is not a deployment decision. Before interpreting or
publishing bifactor scores, a governed workflow should still evaluate:

1. theoretical justification for the general and specific factors;
2. identification and convergence evidence;
3. comparison with correlated-traits, testlet, second-order, or other relevant
   alternatives using relation-appropriate procedures;
4. item and local-dependence diagnostics;
5. parameter and structure recovery under representative designs;
6. DIF, fairness, and stability across relevant populations; and
7. ECV, PUC, omega, and construct-replicability evidence before reporting
   general or domain scores.

No universal cutoff is imposed by the handoff.

## Reference

Gibbons, R. D., & Hedeker, D. R. (1992). Full-information item bi-factor
analysis. *Psychometrika, 57*(3), 423–436.
https://doi.org/10.1007/BF02295430
