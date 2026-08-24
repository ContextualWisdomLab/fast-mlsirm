# Structural selection governor

## Scope

This record governs the next bounded issue #608 slice after the structural
relation contract. Factor retention remains a separate question: retained
primary dimension counts do not select bifactor, higher-order, testlet,
two-tier, many-facet, or latent-space structure.

`fast_mlsirm.structural_selection` consumes explicit
`ModelRelationEvidence`, an already-computed result from the relation-appropriate
comparison procedure, and separately governed recovery and score-interpretation
facts. It performs no likelihood, bootstrap, Vuong, predictive, recovery,
scoreability, or practical-equivalence arithmetic in Python. Numerical
psychometric procedures remain Rust-owned.

## Conservative sequence

The governor preserves this order:

1. establish the actual parameter-space relation rather than infer it from model
   names;
2. run the required regular LR, nonregular/bootstrap LR, or formal Vuong stage;
3. refuse a pairwise winner when distinguishability is absent;
4. require adequate recovery evidence for the candidate family and design;
5. require the intended score interpretation to pass its separate scoreability
   gate; and
6. prefer the simpler candidate only when an already-governed comparison has
   established practical equivalence and that simpler interpretation is
   supported.

The transport does not define a universal practical-equivalence threshold.
That judgement must come from an explicit validation policy appropriate to the
estimand and generalization unit. A caller cannot use a model name, raw
in-sample fit, correlation, or an arbitrary positive log-likelihood variance as
replacement evidence.

## Result semantics

The public result may require relation classification, a likelihood-ratio
stage, formal distinguishability, or Vuong selection before any candidate can
be returned. It can also return `indistinguishable`,
`insufficient_recovery_evidence`, or `score_interpretation_not_supported` rather
than forcing a winner.

When admissible evidence says the candidates are practically equivalent, the
simpler model is preferred only if its intended score interpretation is
supported. If the simpler interpretation fails but the complex interpretation
is supported, the complex candidate may be retained; parsimony never overrides
scoreability.

## Scientific limits

A structural-selection decision is not construct-validity, fairness,
invariance, causal, transportability, or high-stakes deployment evidence.
True-structure recovery should report selection confusion as well as aligned
parameter bias, MAE/RMSE, uncertainty coverage, convergence, and failure
classes. Correlation is supplementary only.

The current Python governor merely preserves already-established evidence and
policy decisions. Future numerical additions for bootstrap tests, formal Vuong
statistics, predictive comparisons, uncertainty, or recovery remain Rust-first
and require their own exact-head scientific evidence.

## References

Preacher, K. J., Zhang, G., Kim, C., & Mels, G. (2013). Choosing the optimal
number of factors in exploratory factor analysis: A model selection perspective.
*Multivariate Behavioral Research, 48*, 28–56.

Rijmen, F. (2010). Formal relations and an empirical comparison among the
bi-factor, the testlet, and a second-order multidimensional IRT model. *Journal
of Educational Measurement, 47*, 361–372.

Rodriguez, A., Reise, S. P., & Haviland, M. G. (2016). Evaluating bifactor
models: Calculating and interpreting statistical indices. *Psychological
Methods, 21*(2), 137–150. https://doi.org/10.1037/met0000045

Schneider, L., Chalmers, R. P., Debelak, R., & Merkle, E. C. (2020). Model
selection of nested and non-nested item response models using Vuong tests.
*Multivariate Behavioral Research, 55*(5), 664–684.
