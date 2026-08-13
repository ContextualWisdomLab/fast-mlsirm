# Structural measurement-model relation contract

## Scope

This record governs the first issue #608 contract that separates **factor
retention** from **structural model comparison**. It classifies a pair of
measurement models only from explicit parameter-space, boundary, constraint,
overlap, and formal distinguishability facts. It does not infer relation from
model-family names and it performs no likelihood, score, information,
bootstrap, or Vuong arithmetic.

The public contract is implemented in `fast_mlsirm.model_relation`. Numerical
comparison procedures remain Rust-owned when implemented.

## Decision

The contract represents these relation states:

- `regular_nested`;
- `boundary_nested`;
- `nonlinear_constraint_nested`;
- `strictly_non_nested`;
- `overlapping`;
- `indistinguishable`; and
- `unknown`.

It maps them conservatively to the only admissible next procedure:

| Relation evidence | Required next procedure |
| --- | --- |
| regular parameter-space embedding | regular likelihood-ratio procedure |
| boundary or unidentified-under-null embedding | parametric-bootstrap/boundary-aware LR |
| nonlinear-constraint embedding | parametric-bootstrap LR in this conservative first contract |
| non-embedded or overlapping, distinguishability untested | formal Vuong distinguishability |
| formally distinguishable non-embedded pair | Vuong selection stage |
| formally indistinguishable pair | no pairwise selection |
| incomplete relation facts | relation classification before testing |

Boundary or unidentified-null evidence takes precedence over a simultaneous
nonlinear-constraint flag so ordinary chi-square is never enabled for the more
nonregular condition.

## Evidence boundary

`ModelRelationEvidence` is a typed transport for already-established structural
facts. This first slice does **not** prove that a caller classified an embedding,
boundary, overlap, or unidentified-null condition correctly. A later fit-family
evidence interface must derive and replay those facts from actual parameter
ordering, constraints, boundary metadata, casewise likelihoods, score vectors,
and information matrices.

The contract rejects contradictory combinations. Nested boundary/constraint
facts cannot be attached to a non-embedded pair; overlap and distinguishability
facts cannot be attached to an embedded pair; and a distinguishability result
cannot be supplied before overlap classification.

## Interpretation boundary

`selection_permitted=True` means only that the relation contract identifies a
scientifically admissible comparison stage. It is not a winner, model-fit result,
validity claim, scoreability claim, recovery result, or deployment approval.

A regular LR remains appropriate only when its regularity assumptions hold.
Boundary, singular, or unidentified-under-null relations require a procedure
whose calibration addresses that nonregularity. Non-nested and overlapping
models require formal distinguishability before an A/B preference. An
indistinguishable or unknown result must not be converted to a forced winner by
AIC, BIC, raw log-likelihood variance, or a model-name heuristic.

## Verification contract

`tests/test_model_relation_contract.py` and
`tests/test_model_relation_contract_edges.py` pin:

- separate factor-retention and model-relation namespaces;
- ordinary LR only for regular nesting;
- bootstrap LR for boundary, unidentified-null, and nonlinear-constraint cases;
- mandatory Vuong distinguishability before non-nested selection;
- explicit no-selection and unknown states;
- exact Boolean facts rather than truthiness;
- rejection of contradictory evidence; and
- conservative boundary precedence.

The tests deliberately do not claim that a formal LR, bootstrap, or Vuong
kernel has been implemented by this contract slice.

## References

Hayashi, K., Bentler, P. M., & Yuan, K.-H. (2007). On the likelihood ratio test
for the number of factors in exploratory factor analysis. *Structural Equation
Modeling, 14*, 505–526.

Rijmen, F. (2010). Formal relations and an empirical comparison among the
bi-factor, the testlet, and a second-order multidimensional IRT model. *Journal
of Educational Measurement, 47*, 361–372.

Schneider, L., Chalmers, R. P., Debelak, R., & Merkle, E. C. (2020). Model
selection of nested and non-nested item response models using Vuong tests.
*Multivariate Behavioral Research, 55*(5), 664–684.

Vuong, Q. H. (1989). Likelihood ratio tests for model selection and non-nested
hypotheses. *Econometrica, 57*(2), 307–333.
