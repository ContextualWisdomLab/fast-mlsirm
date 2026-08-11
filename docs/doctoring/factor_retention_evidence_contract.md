# Governed factor-retention evidence contract

## Scope

This record governs the provider-neutral factor-retention evidence aggregator introduced for issue #608. The aggregator receives **already-computed** candidate factor counts from supported methods and records only method identity plus the bounded positive count. It performs no eigenvalue decomposition, MAP calculation, likelihood evaluation, information-criterion calculation, bootstrap likelihood-ratio statistic, predictive scoring, or structural-model selection.

The supported evidence identities are parallel analysis, Velicer MAP, likelihood/information-criterion evidence, bootstrap likelihood-ratio evidence, predictive evidence, and externally supplied evidence that has already satisfied a caller-owned methodological contract. Duplicate method identities are rejected rather than double-counted as independent evidence.

## Decision semantics

Two or more distinct supported methods that return the same candidate count produce `consensus`. Two or more distinct methods that return different counts produce `disagreement`, with the result retaining the minimum and maximum supported counts as a conservative candidate range and **no forced retained count**. Zero or one method produces `insufficient_evidence`; one method may still define an observed candidate range but does not establish cross-method consensus.

The transport layer caps candidate counts at 10,000 as a fail-closed resource and integrity bound. That ceiling is not a psychometric recommendation and does not imply that very high-dimensional solutions are identified, useful, or scientifically interpretable.

## Scientific boundary

Factor retention and structural model selection are different problems. An eigenvalue- or likelihood-based retained factor count does not by itself select correlated MIRT, bifactor, higher-order, testlet, two-tier, many-facet, or latent-space residual structure. Structural selection requires relation classification, relation-appropriate tests, predictive evidence, residual diagnostics, scoreability/invariance gates, and recovery evidence as specified by issue #608.

No factor-retention method receives universal precedence across continuous, binary, ordinal, multilevel, multiple-membership, or longitudinal designs. Disagreement is therefore represented explicitly instead of converted into a synthetic majority or weighted score in Python.

## Numerical ownership

All numerical factor-retention and model-selection arithmetic remains Rust-owned. Python may validate and marshal evidence and may orchestrate already-computed results, but it must not independently implement or duplicate eigenvalue, likelihood, LR/bootstrap, information-criterion, predictive, score/information, or related psychometric kernels.

## Verification contract

`tests/test_factor_retention_contract.py` requires:

- the dedicated namespace to exist;
- typed closed method identities;
- positive integer and fixed-ceiling candidate validation;
- deterministic evidence ordering;
- duplicate-method rejection;
- conservative consensus, disagreement, and insufficient-evidence semantics; and
- refusal of arbitrary non-package evidence entries.

These tests validate the governance/transport contract only. They do not validate the underlying numerical methods, factor recovery, structural selection, score interpretation, fairness, causal claims, or consequential-decision readiness.

## References

Akaike, H. (1987). Factor analysis and AIC. *Psychometrika, 52*, 317–332.

Hayashi, K., Bentler, P. M., & Yuan, K.-H. (2007). On the likelihood ratio test for the number of factors in exploratory factor analysis. *Structural Equation Modeling, 14*, 505–526.

Preacher, K. J., Zhang, G., Kim, C., & Mels, G. (2013). Choosing the optimal number of factors in exploratory factor analysis: A model selection perspective. *Multivariate Behavioral Research, 48*, 28–56.

Schneider, L., Chalmers, R. P., Debelak, R., & Merkle, E. C. (2020). Model selection of nested and non-nested item response models using Vuong tests. *Multivariate Behavioral Research, 55*(5), 664–684.
