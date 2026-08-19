# ADR-0016: Adopt Angoff delta-plot as the small-sample observed-score DIF screen

Status: **Accepted**
Date: 2026-08-16
Supersedes: none
Superseded by: none

## Context

Protected main already exposes Mantel–Haenszel, logistic-regression, and
SIBTEST observed-score DIF screens, with a governed binary-pilot handoff in
`docs/rubric_dif_pilot_handoff.md`. Those procedures remain the matching-score
and model-based-standardization family documented by Holland and Thayer
(1988), Swaminathan and Rogers (1990), and Shealy and Stout (1993).

A separate Rust/Python path, `mlsirm_core::dif::delta_plot` /
`fast_mlsirm.delta_plot`, implements Angoff's transformed-item-difficulty /
delta-plot procedure with the Magis and Facon normal-approximation threshold.
That path was implemented and tested, but the bibliographic record in the DIF
handoff and architecture ADRs omitted Angoff and Magis and Facon. The gap
made it possible to misread the estimator as an unnamed extra DIF flag or,
worse, as a security control.

The implemented procedure is a psychometric item-screen. It is not a CWE,
OWASP, or NIST control.

## Decision drivers

- Small-sample or sparse two-group binary matrices are common in generated-item
  and LLM-judge pilots where a full matching-score table is unstable.
- The repository already ships a delta-plot kernel; the decision record must
  name the method that kernel implements.
- Fairness and score-use interpretation are governed by AERA, APA, and NCME
  (2014), not by software-security catalogs.

## Ownership and dependency direction

`fast-mlsirm` owns the reusable observed-score DIF kernels and their
interpretation boundaries. Hosted participant/group identity resolution,
consent, and operational fairness policy remain downstream
(`ContextualWisdomLab/psychometrics-commons` or another data-owning host).
This ADR does not change the ADR-0001 repository boundary.

## Decision

Adopt Angoff's delta-plot (Angoff, 1972; Angoff & Ford, 1973), with the Magis
and Facon (2012) threshold options as implemented by
`fast_mlsirm.delta_plot`, as the repository's small-sample observed-score DIF
screen.

1. The implemented transform is the ETS-style delta
   `4 * qnorm(1 - p) + 13` on extreme-adjusted group proportions.
2. Detection uses either the Magis and Facon normal-approximation threshold
   (`threshold="norm"`) or a fixed perpendicular-distance threshold
   (`threshold="fixed"`).
3. Optional IPP1/IPP2/IPP3 purification is part of the accepted screen, not a
   separate product.
4. Mantel–Haenszel, logistic DIF, and SIBTEST remain the distinct
   matching-score / standardization screens. Delta-plot does not replace them
   and is not an alias for them.
5. A flag is a review candidate. It is not a finding of unfairness, bias, or
   invariance.

Method documentation: [`../delta_plot_dif.md`](../delta_plot_dif.md).

## Invariants / acceptance evidence

1. `fast_mlsirm.delta_plot` and `mlsirm_core::dif::delta_plot` accept only
   binary `0`/`1`/`NaN` responses and a two-group `0`/`1` coding.
2. Numeric work is Rust-owned; Python validates and marshals.
3. The kernel either returns a typed result or a documented validation /
   non-convergence error. It does not silently average non-binary values.
4. Documentation and ADRs cite Angoff and Magis and Facon as the method
   basis and do not cite CWE/OWASP/NIST as that basis.

## Non-goals and claims not made

- Not a multiple-group IRT, bifactor, or latent-space DIF model.
- Not a missing-data DIF likelihood. The observed-score kernel may drop `NaN`
  cells per item and group. A missing-data DIF likelihood remains on a
  separately reviewed path. An item-group with no observed cells still
  returns an error.
- Not a security vulnerability scanner or AI-risk control.
- Not a claim that `deltaPlotR` is the scientific oracle. The R package is a
  computational comparison source (Magis & Facon, 2014).

## Consequences and trade-offs

### Benefits

- Small-sample pilots have an explicit, literature-named screen.
- DIF family documentation can distinguish delta-plot from MH/logistic/SIBTEST.
- Fairness language stays attached to testing standards rather than security
  catalogs.

### Costs / risks

- Major-axis distances are an observed-score proxy and can confound impact
  with DIF.
- Purification can fail to converge; callers must inspect `converged`.
- Users may over-interpret flags unless the interpretation boundary is kept
  adjacent to the API.

## Alternatives considered

### Use only Mantel–Haenszel / logistic / SIBTEST

Rejected as the sole observed-score family. Those methods remain implemented,
but they are not the kernel that `delta_plot` runs, and they are less
convenient when the matching-score table is sparse.

### Treat delta-plot as a security or model-risk control

Rejected. The method investigates item-group interaction on an observed-score
plot (Angoff, 1972; Angoff & Ford, 1973). NIST/OWASP/CWE do not define it.

### Promote flags to invariance or fairness determinations

Rejected. AERA, APA, and NCME (2014) require broader validity and fairness
evidence than a single observed-score screen.

## Failure, degraded, and recovery behavior

Invalid shapes, non-binary values, empty groups, empty item-group cells, a
degenerate delta cloud, or purification that flags every item return an
error. Purification that hits `max_iter` without a stable flag set returns
`converged=False` rather than inventing a stable set. Callers must not coerce
those outcomes into "no DIF."

## Security and privacy implications

Group codes used for DIF are purpose-limited sensitive attributes under
ADR-0012. This ADR adds no new credential, network, or provider surface. Do
not document the screen as a CWE/OWASP/NIST control.

## Compatibility, migration, and rollback

Public Python/Rust entry points are unchanged. This ADR records the
scientific identity of an already-shipped kernel. Rollback is a documentation
supersession, not an API removal, unless a later ADR retires the kernel.

## Verification and release evidence

- Rust unit oracles and Python `delta_plot` contract tests on protected main.
- No formula change is authorized by this ADR. A later change to the delta
  transform, threshold, or purification algebra would require a model-design
  update and Rust/Python parity evidence.

## Research and standards basis

Angoff, W. H. (1972, September). *A technique for the investigation of
cultural differences* [Paper presentation]. Annual meeting of the American
Psychological Association, Honolulu, HI, United States.

Angoff, W. H., & Ford, S. F. (1973). Item-race interaction on a test of
scholastic aptitude. *Journal of Educational Measurement, 10*(2), 95–105.
https://doi.org/10.1111/j.1745-3984.1973.tb00787.x

Magis, D., & Facon, B. (2012). Angoff's Delta method revisited: Improving DIF
detection under small samples. *British Journal of Mathematical and
Statistical Psychology, 65*(2), 302–321.
https://doi.org/10.1111/j.2044-8317.2011.02025.x

Magis, D., & Facon, B. (2014). deltaPlotR: An R package for differential item
functioning analysis with Angoff's Delta Plot. *Journal of Statistical
Software, 59*(1), 1–19. https://doi.org/10.18637/jss.v059.c01

American Educational Research Association, American Psychological
Association, & National Council on Measurement in Education. (2014).
*Standards for educational and psychological testing*. American Educational
Research Association.

Adjacent DIF family (not this kernel): Dorans and Kulick (1986), Holland and
Thayer (1988), Swaminathan and Rogers (1990), and Shealy and Stout (1993), as
recorded in [`../rubric_dif_pilot_handoff.md`](../rubric_dif_pilot_handoff.md).

## Follow-ups

Calibrated / multiple-group IRT DIF and missing-data DIF remain separately
reviewed. Do not fold them into this observed-score screen.

## Reversal / supersession conditions

Supersede this ADR if the repository retires `delta_plot`, replaces the
Angoff/Magis–Facon algebra with a different DIF estimand, or elevates
delta-plot flags to operational fairness determinations without additional
validity evidence.
