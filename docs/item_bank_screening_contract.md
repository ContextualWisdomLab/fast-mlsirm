# Governed item-bank candidate screening contract

## Scope

This document records the non-numerical screening boundary for issue #609's
second staged item-bank slice. It does not claim that structural or semantic
screening establishes psychometric calibration, construct validity, fairness,
or production approval. It determines only whether one exact generated item
candidate has complete governed screening evidence before piloting.

The contract binds exact item-content, rubric, blueprint, generation-contract,
and screening-policy fingerprints. Raw prompt, response, source, or provider
content is not required in the canonical screening artifact.

## Required screening dimensions

Every candidate result contains exactly one finding for each dimension:

1. answerability;
2. ambiguity/multiple-answer risk;
3. evidence entailment;
4. distractor quality;
5. semantic redundancy;
6. leakage/memorization risk;
7. fairness/bias risk;
8. adversarial instruction/content risk;
9. expected perturbation-anchor direction; and
10. cost/runtime suitability.

Omitting a dimension is not equivalent to passing it. Each finding requires at
least one immutable evidence fingerprint.

## Fail-closed aggregation

The aggregate result is derived rather than caller-supplied:

- any `fail` finding makes the candidate ineligible for piloting;
- one or more `accepted_with_limitation` findings produce an explicit
  limitation-bearing result and require a descriptive limitation code; and
- `pass` is possible only when every governed dimension passes.

An accepted limitation remains visible provenance. It is not erased by later
scores and is not a psychometric approval decision.

## Architecture boundary

`fast_mlsirm` owns this immutable logical evidence contract. Downstream systems
may persist or workflow these artifacts under their own tenancy,
authorization, retention, and audit controls. Calibration, fit, DIF,
information, linking, uncertainty, drift, and selection arithmetic remain on
existing/future Rust-owned numerical surfaces. This module adds no new
likelihood or statistical calculation.

## Research basis

The contract follows evidence-centered assessment design by separating item
claims, evidence, and operational decisions, and it follows automated-scoring
evaluation guidance by requiring explicit evaluation evidence rather than
promoting generated output directly to operational status. Automatic item
generation research supports structured item-model and evidence review but does
not justify treating generated content as calibrated measurement evidence.

### References (APA 7)

Gierl, M. J., & Lai, H. (2012). The role of item models in automatic item
  generation. *International Journal of Testing, 12*(3), 273–298.
  https://doi.org/10.1080/15305058.2011.635830

Mislevy, R. J., Almond, R. G., & Lukas, J. F. (2003). A brief introduction to
  evidence-centered design. *ETS Research Report Series, 2003*(1), i–29.
  https://doi.org/10.1002/j.2333-8504.2003.tb01908.x

Williamson, D. M., Xi, X., & Breyer, F. J. (2012). A framework for evaluation
  and use of automated scoring. *Educational Measurement: Issues and Practice,
  31*(1), 2–13. https://doi.org/10.1111/j.1745-3992.2011.00223.x
