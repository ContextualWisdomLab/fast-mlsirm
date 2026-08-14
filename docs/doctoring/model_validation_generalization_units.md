# Model-validation generalization units and leakage controls

## Decision

Model validation must preserve the scientific unit implied by the intended generalization claim. A response cell is not an admissible standalone generalization unit. The orchestration contract therefore distinguishes person/system, query/testlet, rater/family, domain/language, cluster/context, and temporal-period units and keeps each declared group inside one validation fold.

This layer is intentionally non-numerical. It does not compute likelihoods, fit indices, bootstrap statistics, factor-retention evidence, or model-selection scores. Those result-affecting calculations remain Rust-owned. Python validates the design metadata that determines which observations may be separated during downstream validation.

## Leakage and identity integrity

Grouped validation can appear leakage-safe while still splitting one scientific unit if caller identifiers are malformed. In particular, blank or presentation-padded identifiers can manufacture artificial groups or folds (`"query-a"` versus `" query-a "`). The contract fails closed rather than silently trimming or accepting those identities: trimming would change caller-declared identity semantics, while accepting them would permit a nominally distinct group to bypass the leakage check.

Caller-controlled identity objects are also a trust boundary. Validation accepts exact built-in strings for group and fold entries rather than arbitrary string subclasses, so a caller cannot execute an overridden normalization callback such as `strip()` during security-sensitive identity classification. Invalid identity objects are rejected with package-owned, non-reflective errors before normalization. This is consistent with OWASP ASVS 5.0.0's use as a secure-development verification baseline and with NIST SSDF 1.1's recommendation to address vulnerability root causes during the software-development lifecycle. ASVS 5.0.0 is the latest stable ASVS release; NIST SP 800-218 Version 1.1 remains the current final SSDF baseline while SSDF Version 1.2 is still a draft.

A usable grouped partition requires at least two scientific groups and at least two folds. Repeated observations from one group are allowed inside a single fold, but one group cannot cross fold boundaries.

Temporal-forward validation is separately constrained. Caller-defined integer period ordinals are accepted only when every training period strictly precedes every validation period. Calendar interpretation remains outside this module; the contract prevents look-ahead without inventing date semantics.

## Scientific scope

The validation-unit decision is part of the broader model-selection workflow in issue #608, where factor retention and structural model selection remain separate questions. Relation-appropriate model comparison, predictive scoring, residual diagnostics, scoreability, invariance/DIF, and true-structure recovery remain downstream evidence layers. A valid split design is necessary evidence hygiene, not evidence that any candidate model is correct.

## Verification contract

- scalar strings cannot masquerade as identity vectors;
- blank and surrounding-whitespace group/fold identities fail closed;
- string subclasses with caller-defined normalization callbacks fail closed before callback execution;
- a declared group appearing in multiple folds fails closed;
- a one-group or one-fold partition is rejected as degenerate validation evidence;
- temporal-forward windows reject overlap or look-ahead;
- validation-unit orchestration adds no psychometric numerical arithmetic to Python.

## References

National Institute of Standards and Technology. (2022). *Secure Software Development Framework (SSDF) Version 1.1: Recommendations for mitigating the risk of software vulnerabilities* (NIST Special Publication 800-218). https://doi.org/10.6028/NIST.SP.800-218

OWASP Foundation. (2025). *OWASP Application Security Verification Standard (ASVS) 5.0.0*. https://owasp.org/www-project-application-security-verification-standard/

Preacher, K. J., Zhang, G., Kim, C., & Mels, G. (2013). Choosing the optimal number of factors in exploratory factor analysis: A model selection perspective. *Multivariate Behavioral Research, 48*(1), 28–56. https://doi.org/10.1080/00273171.2012.710386

Schneider, L., Chalmers, R. P., Debelak, R., & Merkle, E. C. (2020). Model selection of nested and non-nested item response models using Vuong tests. *Multivariate Behavioral Research, 55*(5), 664–684. https://doi.org/10.1080/00273171.2019.1664280

Rijmen, F. (2010). Formal relations and an empirical comparison among the bi-factor, the testlet, and a second-order multidimensional IRT model. *Journal of Educational Measurement, 47*(3), 361–372. https://doi.org/10.1111/j.1745-3984.2010.00118.x
