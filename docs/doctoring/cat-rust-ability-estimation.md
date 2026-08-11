# CAT Rust-owned ability-estimation doctoring

## Status and scope

This record governs the computerized adaptive testing (CAT) ability-estimation
boundary in `python/fast_mlsirm/cat.py` and the compiled Rust APIs in
`crates/mlsirm-core/src/scoring.rs`. The calibrated item bank remains the source
of item parameters. Python validates public inputs, normalizes arrays, and owns
the adaptive-test policy; Rust owns probability evaluation, MLE Newton updates,
EAP posterior moments, and information-based standard-error reduction.

The supported CAT parameterization is simple structure: each item maps to one
trait through `factor_id`. That makes the current MLE, EAP, and information
reductions block-separable by trait dimension. It is an explicit operational
boundary, not a claim that a fully unrestricted multidimensional CAT is
implemented. The repository's calibration and marginal APIs retain their
multigroup, multilevel, and latent-population responsibilities; this CAT slice
must not silently reinterpret those population structures as examinee-level
independent traits.

## Numerical and execution contract

- MLE uses a bounded Newton update with Fisher information curvature. A
  dimension with an all-correct or all-incorrect administered response pattern
  is returned with `finite=False` and infinite standard error because its
  unregularized likelihood has no finite root.
- EAP evaluates a prior-centred fixed grid and returns posterior mean and
  standard deviation. A dimension with no administered items returns the prior
  moments exactly.
- Standard errors are computed from the existing Rust bank-information
  reduction, with the explicit device contract preserved: `auto` may use the
  repository's GPU path and otherwise falls back to deterministic Rust CPU
  reduction. MLE and EAP dimension workers use scoped Rust CPU threads with
  independent dimension state; no Python numerical loop remains on the public
  estimation path.
- The latent-space position is the calibrated bank population mean, matching
  the existing CAT information and probability convention. This is recorded so
  a future directional-information or person-specific latent-position feature
  cannot be introduced as an accidental semantic change.

## Evidence and realistic tests

The ownership tests replace each compiled entry point with a sentinel and prove
that the public Python functions do not recompute the result. Rust tests cover
mixed-response finite MLE estimates, perfect-pattern non-finite flags, prior
preservation with no administered items, and administered-item information
masks. The Python CAT suite runs seeded adaptive administrations against a
known true trait, checks mean EAP and MLE recovery over repeated Bernoulli
responses, verifies decreasing standard error, checks the MLE score equation,
and confirms finite EAP behavior for extreme response patterns.

The evidence supports numerical ownership and the tested operational
parameterization. It does not by itself establish construct validity, fairness,
diagnostic use, causal utility, or readiness for high-stakes decisions.

## Rollback and release boundary

If a compiled-backend defect is found, disable the affected public CAT entry
point or revert to the last verified Rust implementation while retaining the
ownership and true-parameter recovery tests. Do not restore a silent Python
numerical fallback: that would make the device and audit contract ambiguous.
Any future multidimensional, longitudinal, or multilevel CAT extension must
define its estimand, time/provenance fields, identification constraints,
simulation recovery evidence, and APA 7th source traceability before changing
this boundary.

## References

Bock, R. D., & Mislevy, R. J. (1982). Adaptive EAP estimation of ability in a
microcomputer environment. *Applied Psychological Measurement, 6*(4),
431–444. https://doi.org/10.1177/014662168200600405

Gibbons, R. D., & Hedeker, D. R. (1992). Full-information item bi-factor
analysis. *Psychometrika, 57*(3), 423–436.
https://doi.org/10.1007/BF02295430

Lord, F. M. (1980). *Applications of item response theory to practical testing
problems*. Lawrence Erlbaum Associates.

van der Linden, W. J., & Pashley, P. J. (2010). Item selection and ability
estimation in adaptive testing. In W. J. van der Linden & C. A. W. Glas (Eds.),
*Elements of adaptive testing* (pp. 3–30). Springer.
https://doi.org/10.1007/978-0-387-85461-8_1

Warm, T. A. (1989). Weighted likelihood estimation of ability in item response
theory. *Psychometrika, 54*(3), 427–450. https://doi.org/10.1007/BF02294627

Weiss, D. J., & Kingsbury, G. G. (1984). Application of computerized adaptive
testing to educational problems. *Journal of Educational Measurement, 21*(4),
361–375. https://doi.org/10.1111/j.1745-3984.1984.tb01040.x
