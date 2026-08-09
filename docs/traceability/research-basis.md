# Research-to-architecture basis

Status: **Authoritative research traceability baseline**  
Last reviewed: 2026-08-09

This document records why major scientific/product directions exist. It does not promote every research idea to accepted functionality. `Accepted`, `Proposed`, and `Open` below match the ADR status/implementation evidence.

## 1. MLSIRM / MLS2PLM numerical core — Accepted

Architecture effect:

- retain the existing simple-structure MLS2PLM as an explicit specialization rather than silently claiming the full discrimination-vector model;
- preserve latent-distance interaction as a residual person-item interaction construct;
- use identification-aware recovery/parity for latent coordinates.

Primary basis:

- Jeon, M., Jin, I. H., Schweinberger, M., & Baugh, S. (2021). Mapping unobserved item-respondent interactions: A latent space item response model with interaction map. *Psychometrika, 86*(2), 378–403.
- Kang, I., & Jeon, M. (2025). Multidimensional latent space item response models: A note on the relativity of conditional dependence. *Psychometrika, 90*(2), 799–826.
- Molenaar, D., & Jeon, M. (2026). Regularized joint maximum likelihood estimation of latent space item response models. *Psychometrika, 91*, 335–359.

## 2. Reference-free RAG as measurement — Proposed

Architecture effect:

- RAGAS/LLM-judge outputs are observations, not truth;
- separate groundedness, correctness, retrieval relevance/coverage, utility/completeness, robustness, abstention and citation attribution;
- retain query/testlet, judge family/model, prompt/occasion and system-run identities;
- calibrate judges/facets before interpreting aggregate system quality;
- use multidimensional/bifactor/testlet structure before residual latent-space interaction.

Research basis:

- Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2024). RAGAS: Automated evaluation of retrieval augmented generation. *Proceedings of EACL 2024*.
- Saad-Falcon, J., Khattab, O., Potts, C., & Zaharia, M. (2024). ARES: An automated evaluation framework for retrieval-augmented generation systems. *Proceedings of NAACL 2024*.
- Jeon/Kang latent-space work above for residual interaction structure.
- Many-facet measurement literature for evaluator severity and design connectedness.

Open question: a canonical end-to-end RAG observation schema and Rust joint estimator are not yet accepted product behavior.

## 3. Multidimensional, bifactor, testlet and latent-space hierarchy — Accepted decision rule / evolving implementation

Architecture effect:

- correlated substantive dimensions, a possible general bifactor dimension, testlet/local-dependence effects and latent-space residual interactions are complementary layers rather than substitutes;
- bifactor fit does not automatically authorize general/specific scores;
- latent space is added only after known substantive/facet/testlet structure when held-out/recovery evidence supports it.

Primary basis:

- Rodriguez, A., Reise, S. P., & Haviland, M. G. (2016). Evaluating bifactor models: Calculating and interpreting statistical indices. *Psychological Methods, 21*, 137–150.
- Rijmen, F. (2010). Formal relations and an empirical comparison among the bi-factor, the testlet, and a second-order multidimensional IRT model. *Journal of Educational Measurement, 47*, 361–372.
- Cai, L. (2010). A two-tier full-information item factor analysis model with applications. *Psychometrika, 75*, 581–612.
- Kang & Jeon (2025) for conditional-dependence relativity.

## 4. Relation-safe model selection — Accepted

Architecture effect:

- determine factor retention separately from structural model choice;
- determine nestedness/overlap from actual constraints;
- formal distinguishability precedes non-nested preference;
- boundary/singular nulls require boundary-aware/bootstrap evidence;
- cluster-aware held-out prediction and true-structure recovery supplement inferential tests.

Primary basis:

- Schneider, L., Chalmers, R. P., Debelak, R., & Merkle, E. C. (2020). Model selection of nested and non-nested item response models using Vuong tests. *Multivariate Behavioral Research, 55*, 664–684.
- Preacher, K. J., Zhang, G., Kim, C., & Mels, G. (2013). Choosing the optimal number of factors in exploratory factor analysis: A model selection perspective. *Multivariate Behavioral Research, 48*, 28–56.
- Fujimoto, K. A., & Falk, C. F. (2024). The accuracy of Bayesian model fit indices in selecting among multidimensional item response theory models. *Educational and Psychological Measurement, 84*, 217–244.

## 5. Dynamic evidence-grounded rubric/item bank — Proposed

Architecture effect:

- rubric -> blueprint -> candidate generation -> screening -> artificial crowd -> calibration -> adaptive/governed item bank -> rubric revision;
- benchmark criteria are candidate-blind; candidate-aware discovery is cross-fitted/separate;
- criteria carry evidence regime, provenance, version/lifecycle identity;
- structural schema validity is distinct from construct/content validity;
- item information and fit replace ad hoc LLM weights after calibration.

Research basis:

- Hashemi et al. (2024), LLM-Rubric and calibrated multidimensional evaluation.
- AutoNuggetizer/TREC RAG work for generated atomic evaluation obligations.
- Shankar et al. (2024), EvalGen, for criteria drift and validator alignment.
- emerging 2025–2026 instance-specific/recursive rubric and automated item-generation work is treated as promising but not settled evidence; implementation claims remain conservative.

## 6. Automated essay scoring / automated scoring as many-facet measurement — Accepted baseline / evolving

Architecture effect:

- scorer output is governed observation evidence, not an oracle score;
- human and AI raters share a calibration framework;
- severity, agreement, range use, DIF/fairness, drift and adjudication are separate evidence dimensions;
- raw human-AI correlation is insufficient as the primary validity claim.

Primary basis:

- Williamson, D. M., Xi, X., & Breyer, F. J. (2012). A framework for evaluation and use of automated scoring. *Educational Measurement: Issues and Practice, 31*(1), 2–13.
- Uto, M., & Ueno, M. (2020). A generalized many-facet Rasch model and its Bayesian estimation using Hamiltonian Monte Carlo. *Behaviormetrika, 47*, 469–496.
- AERA, APA, & NCME (2014), *Standards for Educational and Psychological Testing*.

## 7. Parameter recovery over correlation — Accepted

Architecture effect:

- align scale first;
- require bias, MAE/RMSE, SE/interval coverage, convergence and function/information recovery where relevant;
- correlation is supplementary only.

Primary basis:

- Svetina, D., Valdivia, A., Underhill, S., Dai, S., & Wang, X. (2017). Parameter recovery in multidimensional item response theory models under complexity and nonnormality. *Applied Psychological Measurement, 41*(7), 530–544.
- Bland, J. M., & Altman, D. G. (1986). Statistical methods for assessing agreement between two methods of clinical measurement. *The Lancet, 327*(8476), 307–310.

## 8. Multilevel, multiple-membership and temporal measurement — Proposed

Architecture effect:

- prevent atomistic flattening;
- explicit context dimensions and weighted memberships;
- separate repeated occasion ordering from continuous-time dynamics;
- require Rust estimator identification/recovery before production claims.

Primary basis:

- Fox, J.-P., & Glas, C. A. W. (2001). Bayesian estimation of a multilevel IRT model. *Psychometrika, 66*, 271–288.
- Uto, M. (2022). A Bayesian many-facet Rasch model with Markov modeling for rater severity drift. *Behavior Research Methods, 55*, 3910–3928.

## 9. Adaptive factor rotation — Proposed

Architecture effect:

- no universal-best criterion;
- criterion registry separated from optimizer;
- deterministic multi-start and solution-basin diagnostics;
- common empirical selector using stability/recovery/theory rather than raw cross-criterion objective values.

Primary basis:

- Bernaards, C. A., & Jennrich, R. I. (2005). Gradient projection algorithms and software for arbitrary rotation criteria in factor analysis. *Educational and Psychological Measurement, 65*(5), 676–696.

## 10. Enterprise issue measurement — reusable adapter accepted; causal decision layer outside current measurement core

Architecture effect:

- evidence, counterevidence, stakeholder perspective and candidate intervention are preserved distinctly;
- criterion observations feed the shared scoring/facets architecture;
- a latent measurement score is not itself an expected business intervention value;
- causal outcome/utility optimization belongs in a separate decision layer or downstream bounded context unless a reusable decision-theory primitive is explicitly added.

The long-term decision concept discussed in research is expected net intervention value, but it is not an accepted `fast-mlsirm` psychometric-kernel requirement today.

## 11. LLM orchestration depth — Accepted governance, open product-specific policy

Architecture effect:

- use provider-neutral orchestration and NVIDIA NIM credentials where needed;
- preserve deterministic no-model gates;
- compare simple routing versus deeper agent orchestration under comparable budgets before hard-coding a complex topology;
- treat Fugu/Conductor/TRINITY-class results as research input, not a universal mandate.

## 12. Standards baseline

Architecture/product documentation also uses:

- ISO/IEC/IEEE 29148:2018 for requirements engineering;
- ISO/IEC/IEEE 42010:2022 for architecture-description concerns/viewpoints;
- ISO/IEC 25010:2023 for product-quality concerns;
- ISO/IEC 42001:2023 for AI lifecycle/governance concerns where applicable;
- WCAG 2.2 for accessible report surfaces;
- AERA/APA/NCME 2014 Testing Standards for evidence, fairness and score-use interpretation.

## Maintenance rule

When a new research finding changes a released formula, score interpretation, factor/model relation, validity gate, item/rubric lifecycle or rater-evaluation rule, update the relevant ADR, requirement IDs and this traceability record in the same reviewed change.
