# Doctoring record: explicit decision-support expected values

## Purpose

This record documents the first decision-support slice for issue #404. It adds
one provider-neutral, finite-table numerical boundary without adding an LLM
provider, a parallel enterprise observation schema, or a causal model.

`fast_mlsirm.decision_support.evaluate_decision_support()` accepts:

- a prior probability vector over finite states;
- an action-by-state utility table supplied by the caller;
- non-negative intervention costs and an explicit no-action row; and
- optionally, a joint signal-by-state probability table for sample information.

The Python adapter validates shape, storage, and control identity, then
marshals the arrays. Expected-value arithmetic is implemented in the Rust core.

## Equation-to-source traceability

For action `a` and state `s`, the implementation defines the net intervention
value as:

```text
N(a,s) = U(a,s) - U(no_action,s) - cost(a)
```

The prior expected value for each action is `sum_s p(s) N(a,s)`. The selected
action is the first row attaining the maximum, making exact ties deterministic.
When the state is known before acting, the value is

```text
EVPI = sum_s p(s) max_a N(a,s) - max_a sum_s p(s) N(a,s)
```

For a caller-supplied joint distribution `p(z,s)`, the expected value after a
sample signal is evaluated without dividing by a zero-probability signal:

```text
EVSI = sum_z max_a sum_s p(z,s) N(a,s) - max_a sum_s p(s) N(a,s)
```

The reported net sample-information value subtracts the explicit information
cost. The Rust boundary requires the joint table's state marginal to agree
with the prior, so arbitrary posterior rows cannot be presented as coherent
sample information.

Howard (1966) establishes that information value must combine uncertainty with
the consequences of decisions, rather than probability alone. Raiffa and
Schlaifer (1961) provide the expected-utility, perfect-information, and
sample-information decision-analysis framework used here. Their books are
copyrighted and are cited and linked rather than redistributed in this
repository.

## Scope and non-claims

Utilities and probabilities are explicit caller inputs; this module does not
learn organizational preferences, infer probabilities from sentiment or
source text, estimate intervention effects, or claim that an intervention is
causal. The result is not a validity, fairness, calibration, or high-stakes
deployment decision. Queue routing, urgency under delay, stakeholder-specific
views, and human-anchored policy evaluation remain later bounded slices of
issue #404.

The finite-table boundary also deliberately does not accept an arbitrary set of
posterior distributions. A joint state/signal table is required so that EVSI is
auditable and its marginal can be replayed against the prior.

## Verification

Rust tests cover expected-value and EVPI identities, coherent EVSI, zero-mass
signals, explicit no-action baselines, deterministic ties, non-finite utility
contrasts, dimension limits, probability mass, and marginal consistency. Python
tests cover result marshalling, optional information, shape and control
validation, hostile boolean input, and the Rust call boundary.

No likelihood, gradient, Hessian, psychometric estimator, or existing MLSIRM /
MLS2PLM formula is changed by this slice.

## References

Howard, R. A. (1966). Information value theory. *IEEE Transactions on Systems
Science and Cybernetics, 2*(1), 22–26. https://doi.org/10.1109/TSSC.1966.300074

Raiffa, H., & Schlaifer, R. (1961). *Applied statistical decision theory*.
Division of Research, Graduate School of Business Administration, Harvard
University. https://books.google.com/books?id=SpO0KFcFQDsC
