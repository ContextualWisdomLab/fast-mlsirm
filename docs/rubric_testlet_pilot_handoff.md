# Generated-item pilot handoff to binary testlet calibration

`fast_mlsirm.rubric.build_testlet_pilot_design` converts replay-verified
binary pilot observations into a deterministic, content-addressed handoff for
the existing Rust-backed `fast_mlsirm.fit_testlet` API.

The handoff is deliberately narrow. It preserves the governed response and
provenance boundary and discloses the item-to-testlet mapping; it does not
claim that local dependence exists, that the testlet model fits adequately,
or that fitted item or person estimates are fair, scoreable, or valid for a
high-stakes use.

## Assignment contract

Every generated item already carries a descriptive `query_testlet_id` in its
admitted pilot provenance. The handoff:

- retains sorted descriptive testlet identifiers in `testlet_ids`;
- maps every item to exactly one zero-based transport identifier in
  `item_testlet_ids`;
- exposes `testlet_item_counts`, `multi_item_testlet_ids`, and
  `singleton_testlet_ids`; and
- binds the mapping and complete nested binary design into one SHA-256
  fingerprint.

The integer vector is only the representation consumed by the numerical API.
The descriptive identifiers, rubric-generation provenance, response states,
and per-cell rater assignments remain available in the audit artifact.

Singleton testlets are preserved rather than dropped. The existing testlet
fitter fixes their variance to zero because a within-testlet dependence
variance is not identified from one item. A design containing singletons is
therefore disclosed explicitly rather than silently relabeled as evidence of
local dependence.

## Example

```python
from fast_mlsirm import fit_testlet
from fast_mlsirm.rubric import build_testlet_pilot_design

# records are PilotObservationRecord objects issued from admitted pilot items.
design = build_testlet_pilot_design(records)

fit = fit_testlet(
    **design.to_fit_testlet_kwargs(),
    model="rasch",
    q_gamma=21,
    require_convergence=True,
)
```

`to_fit_testlet_kwargs()` supplies:

- a fresh `float64` persons-by-items response matrix;
- `NaN` for missing, not-applicable, and insufficient-evidence cells; and
- a fresh per-item zero-based testlet assignment vector.

The method does not choose between the Rasch and 2PL testlet specifications or
set numerical controls. Those remain explicit caller decisions and are
validated by `fit_testlet`; the handoff does not imply that either
specification is universally preferable.

## Reused fail-closed boundary

The builder delegates response assembly to `build_mirt_pilot_design`, so it
inherits the same contracts:

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

## Required downstream gates

A successful handoff is not a deployment decision. Before interpreting a
fitted testlet model, a governed workflow should still evaluate:

1. a substantive reason that bundled items share a stimulus or method effect;
2. design connectedness, convergence, and boundary behavior of variance
   estimates;
3. comparison with an ordinary conditional-independence model and other
   theoretically relevant structures using relation-appropriate procedures;
4. parameter recovery and false-positive behavior under representative sample
   sizes, bundle sizes, missingness, and local-dependence strengths;
5. item fit, residual local dependence, DIF, fairness, and stability across
   relevant populations; and
6. the consequences of treating singleton bundles as zero-variance testlets.

A large estimated testlet variance is evidence about residual bundle
dependence under the fitted model, not proof of a causal stimulus effect or of
score validity.

## References

Bradlow, E. T., Wainer, H., & Wang, X. (1999). A Bayesian random effects model
for testlets. *Psychometrika, 64*(2), 153–168.
https://doi.org/10.1007/BF02294533

Wang, X., Bradlow, E. T., & Wainer, H. (2002). A general Bayesian model for
testlets. *Applied Psychological Measurement, 26*(1), 109–128.
https://doi.org/10.1177/0146621602026001007
