# ADR-0005: Human and automated scorers are fallible raters

Status: **Accepted**  
Date: 2026-08-09

## Context

Automated-scoring and LLM-as-a-Judge systems are often validated by correlating machine scores with one human score or by averaging multiple judges. That can hide rater severity, range compression, criterion bias, drift, shared shortcuts and human measurement error. A high correlation preserves rank but does not prove agreement, calibration, fairness or true-parameter recovery.

The protected codebase already contains governed scoring contracts, essay/enterprise adapters and Rust-backed criterion many-facet calibration/reporting paths.

## Decision

Human, LLM and external automated scorers are represented as raters/engines producing observations under explicit assessment/rubric/task revisions. No rater is automatically the truth source.

The scoring architecture must preserve:

- rater/engine identity and revision;
- criterion and rubric identity;
- task and exact task revision;
- respondent/system-run identity;
- response artifact/revision identity;
- terminal observation state;
- evidence references where the scoring contract requires them;
- prompt/occasion/model version where an LLM judge is used.

Many-facet calibration is the baseline mechanism for separating respondent/person, task/item and rater severity effects when the design identifies them. Future generalized rater discrimination, criterion-specific bias, range restriction and time-varying severity/drift require explicit model-identification and recovery evidence before production release.

## Validation evidence

Correlation may be shown as supplementary association evidence. Acceptance decisions use appropriate combinations of:

- QWK, exact and adjacent agreement for ordinal ratings;
- absolute error/bias where a defensible reference scale exists;
- rater severity and fit;
- paired range/dispersion evidence;
- DIF/subgroup error and invariance evidence;
- human-human degradation/comparator evidence when available;
- true-parameter recovery in simulation;
- drift/retest stability across model/prompt/occasion revisions.

`abstained`, `failed`, `excluded` and `scored` states remain distinct. Abstention or infrastructure failure is not converted to the lowest content score.

## Identification invariants

- The respondent/person axis must represent the entity whose latent property is interpreted; response IDs do not substitute for that entity when repeated tasks exist.
- Respondent-task and task-rater graphs must be connected enough for the effects being estimated.
- Multiple raters may score the same exact response revision.
- One respondent-task cell cannot silently bind multiple response revisions.

## Consequences

This supports defensible automated essay scoring, enterprise issue evaluation and LLM judge calibration. It also means a simple scorer wrapper cannot be declared validated merely because average agreement is high.

## Alternatives considered

- **Treat consensus/majority vote as truth.** Rejected because correlated rater errors and severity remain hidden.
- **Use one expert human as gold.** Rejected as the default scientific model; human anchors can be valuable but should retain rater uncertainty unless independently established as an authoritative answer key.
- **Use raw machine-human correlation as the primary gate.** Rejected because correlation is insensitive to additive/scale bias and depends on sample heterogeneity.

## Research and standards basis

This ADR is about rater/score interpretation, agreement, and fairness evidence. NIST, OWASP, and CWE catalogs are not the methodological basis.

American Educational Research Association, American Psychological Association, & National Council on Measurement in Education. (2014). *Standards for educational and psychological testing*. American Educational Research Association.

Bland, J. M., & Altman, D. G. (1986). Statistical methods for assessing agreement between two methods of clinical measurement. *The Lancet, 327*(8476), 307–310. https://doi.org/10.1016/S0140-6736(86)90837-8

Uto, M., & Ueno, M. (2020). A generalized many-facet Rasch model and its Bayesian estimation using Hamiltonian Monte Carlo. *Behaviormetrika, 47*, 469–496. https://doi.org/10.1007/s41237-020-00115-7

Williamson, D. M., Xi, X., & Breyer, F. J. (2012). A framework for evaluation and use of automated scoring. *Educational Measurement: Issues and Practice, 31*(1), 2–13. https://doi.org/10.1111/j.1745-3992.2011.00223.x
