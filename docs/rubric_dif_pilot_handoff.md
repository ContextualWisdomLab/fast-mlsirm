# Generated-item pilot handoff to observed-score DIF screening

`DifPilotDesign` is the governed bridge from replay-verified binary pilot
observations to the repository's Rust-backed observed-score differential item
functioning (DIF) procedures. It binds an immutable binary pilot design to an
explicitly named reference group and focal group without changing responses,
selecting cases, aggregating raters, or performing psychometric arithmetic.

## Contract

`build_dif_pilot_design(records, respondent_groups=..., reference_group_id=...,
focal_group_id=...)` first delegates to `build_mirt_pilot_design`. The existing
binary assembler remains the source of truth for:

- one pilot-study identity;
- complete per-item generation, rubric, query/testlet, judge-policy, occasion,
  and admitted-pilot provenance;
- one response per respondent-item cell;
- binary observed categories without silent dichotomization;
- explicit `missing`, `not_applicable`, and `insufficient_evidence` states;
- retained per-cell rater provenance;
- observed support for every indexed respondent and item; and
- bounded persons-by-items allocation.

The DIF handoff then requires exactly one group assignment for every indexed
respondent. Group identifiers are descriptive nonnumeric identifiers rather
than bare `0`/`1` labels. The declared reference and focal identifiers, the
ordered respondent assignments, and the complete nested binary design are all
included in the SHA-256 design fingerprint. `group_array()` exposes a copied
numeric vector only at the estimator boundary, using the documented convention
`0 = reference` and `1 = focal`.

## Missingness boundary

The current `mantel_haenszel_dif`, `logistic_dif`, purified variants, and
`sibtest` interfaces require a complete binary persons-by-items matrix. The
governed design preserves every non-observed state in its nested
`MirtPilotDesign`, but `to_observed_score_dif_kwargs()` rejects the first
non-observed cell with a structured `dif_incomplete_response_matrix` error.
It never performs complete-case deletion, itemwise deletion, imputation, or
failure-score coercion.

A caller that needs a missing-data DIF model must use a separately reviewed
model whose likelihood and missingness assumptions are explicit; it must not
reinterpret this observed-score handoff as such a model.

## Interpretation boundary

A successful handoff establishes only that the group contract and complete
binary response matrix can be reconstructed reproducibly for existing DIF
screening functions. It does not show that:

- a flagged item is unfair or causally biased;
- an unflagged item is invariant;
- the matching score is uncontaminated;
- impact and DIF are separated adequately;
- purified p-values retain nominal false-discovery control;
- the sample has adequate focal/reference support;
- the item bank is scoreable, reliable, or valid; or
- generated items are suitable for operational or high-stakes use.

Observed-score procedures should be triangulated with theory, descriptive group
support, calibrated DIF where justified, sensitivity to anchor selection,
subgroup and intersectional analyses, recovery simulation, and human review.
The generated-item audit and schema gates remain governance boundaries rather
than evidence of measurement invariance.

The Angoff transformed-item-difficulty / delta-plot screen is part of this
observed-score DIF family. It is documented separately in
[`delta_plot_dif.md`](delta_plot_dif.md) and decided in
[`adr/0018-angoff-delta-plot-dif.md`](adr/0018-angoff-delta-plot-dif.md). It is
not a CWE, OWASP, or NIST control, and it is not an alias for Mantel–Haenszel,
logistic DIF, or SIBTEST.

## Primary methodological sources

Dorans, N. J., & Kulick, E. (1986). Demonstrating the utility of the
standardization approach to assessing unexpected differential item performance
on the Scholastic Aptitude Test. *Journal of Educational Measurement, 23*(4),
355–368. https://doi.org/10.1111/j.1745-3984.1986.tb00255.x

Holland, P. W., & Thayer, D. T. (1988). Differential item performance and the
Mantel-Haenszel procedure. In H. Wainer & H. I. Braun (Eds.), *Test validity*
(pp. 129–145). Erlbaum.

Swaminathan, H., & Rogers, H. J. (1990). Detecting differential item functioning
using logistic regression procedures. *Journal of Educational Measurement,
27*(4), 361–370. https://doi.org/10.1111/j.1745-3984.1990.tb00754.x

Shealy, R., & Stout, W. (1993). A model-based standardization approach that
separates true bias/DIF from group ability differences and detects test bias/DTF
as well as item bias/DIF. *Psychometrika, 58*(2), 159–194.
https://doi.org/10.1007/BF02294572

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
