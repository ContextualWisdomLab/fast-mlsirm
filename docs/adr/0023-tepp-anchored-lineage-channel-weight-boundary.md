# ADR-0023: TEPP-anchored Event Lineage channel-weight boundary

Status: **Proposed**
Date: 2026-08-26

## Context

LineageWeave ADR 0208 assigns psychometric channel-weight arithmetic to this
repository and requires the accepted TEPP independent anchor from its ADR 0205.
The legacy consumer converts continuous channel scores into binary responses at
a locally selected fusion floor, fits an internally anchored MLS2PLM, and
normalizes expected information in Python. That floor determines the responses
and therefore the fitted weights. It is not an independent criterion and cannot
be migrated as a scientifically equivalent Rust implementation.

TEPP's published `tepp.lineage_criterion_anchor.v1` is a validity decision for
an already proposed estimation run. It binds run, snapshot, cutoff, accepted or
rejected status, and validated pair count, but deliberately transports no
pair-level independent criterion observations. The accepted flag cannot be
used as the response for every pair; doing so would manufacture the missing
criterion and make parameter-recovery tests circular.

## Decision

The Rust core admits `fast-mlsirm.lineage_channel_weight_evidence.v1`, which
binds a complete continuous pair-by-channel matrix to one estimation run,
snapshot, cutoff, reconstruction groups, and the exact accepted TEPP v1
projection. Unknown fields, mixed identities, duplicate pairs or channels,
non-finite/out-of-range scores, rejected anchors, and pair-count mismatches fail
closed.

The public estimation entry point returns
`IndependentCriterionObservationsUnavailable`. It performs no score
dichotomization, IRT fit, expected-information normalization, convex fusion, or
ranking. In particular, TEPP acceptance is necessary for activation but is not
sufficient input to reproduce or estimate weights.

A future estimator requires a producer-owned successor that binds an
independent pair-level criterion posterior or outcomes, their sampling/design
provenance, and their exact pair identities to the same run/snapshot/cutoff.
Before release, its model and weight functional require a separate accepted ADR,
Rust CPU and GPU implementations of the same objective, true-parameter and
known-weight recovery, uncertainty coverage, and a method-derived parity bound.
No channel floor, keyword, rank, equal share, hand-selected prior, or consumer
normalization may fill the missing producer evidence.

## Consequences

- LineageWeave remains fail closed and may delete its Python estimator only
  after the successor producer and Rust estimator are accepted.
- The v1 Rust contract provides an immutable consumer seam without claiming a
  calibrated weight vector.
- Period-report calibration, expected-response aggregation, interaction maps,
  and channel weighting remain separate owner contracts and PR stacks.

## References

American Educational Research Association, American Psychological Association,
& National Council on Measurement in Education. (2014). *Standards for
educational and psychological testing*. American Educational Research
Association.

Birnbaum, A. (1968). Some latent trait models and their use in inferring an
examinee's ability. In F. M. Lord & M. R. Novick, *Statistical theories of
mental test scores* (pp. 397–479). Addison-Wesley.

Fox, J.-P., & Glas, C. A. W. (2001). Bayesian estimation of a multilevel item
response theory model using Gibbs sampling. *Psychometrika, 66*(2), 271–288.
https://doi.org/10.1007/BF02294839
