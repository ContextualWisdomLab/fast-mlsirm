# Research-to-architecture basis

Status: **Authoritative research traceability baseline**  
Last reviewed: 2026-08-16

This document records why major scientific/product directions exist. It does not promote every research idea to accepted functionality. `Accepted`, `Proposed`, and `Open` below match the ADR status/implementation evidence.

## 1. MLSIRM / MLS2PLM numerical core — Accepted

Architecture effect:

- retain the existing simple-structure MLS2PLM as an explicit specialization rather than silently claiming the full discrimination-vector model;
- preserve latent-distance interaction as a residual person-item interaction construct;
- use identification-aware recovery/parity for latent coordinates.

Primary basis:

- Jeon, M., Jin, I. H., Schweinberger, M., & Baugh, S. (2021). Mapping unobserved item-respondent interactions: A latent space item response model with interaction map. *Psychometrika, 86*(2), 378–403. https://doi.org/10.1007/s11336-021-09762-5
- Kang, I., & Jeon, M. (2025). Multidimensional latent space item response models: A note on the relativity of conditional dependence. *Psychometrika, 90*(2), 799–826. https://doi.org/10.1017/psy.2025.5
- Molenaar, D., & Jeon, M. (2026). Regularized joint maximum likelihood estimation of latent space item response models. *Psychometrika, 91*, 335–359. https://doi.org/10.1017/psy.2025.10068
- American Educational Research Association, American Psychological Association, & National Council on Measurement in Education. (2014). *Standards for educational and psychological testing*. American Educational Research Association.

Decision record: ADR-0001. Equation contract: `docs/papers/mls2plm-canonical-equations.md`.

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

- Hashemi, H., Eisner, J., Rosset, C., Van Durme, B., & Kedzie, C. (2024). LLM-RUBRIC: A multidimensional, calibrated approach to automated evaluation of natural language texts. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 13806–13834). Association for Computational Linguistics. https://aclanthology.org/2024.acl-long.745/
  Scope: calibrates multiple rubric-question outputs to human annotations and motivates treating automated judges as fallible, multidimensional raters rather than truth oracles.
- Pradeep, R., Thakur, N., Upadhyay, S., Campos, D., Craswell, N., & Lin, J. (2024). *Initial nugget evaluation results for the TREC 2024 RAG track with the AutoNuggetizer framework* (arXiv preprint arXiv:2411.09607). https://arxiv.org/abs/2411.09607
  Scope: describes atomic information nuggets, human post-editing/calibration and semantic assignment for reference-free RAG evaluation; it does not justify lexical-only matching or an absolute-truth claim.
- Shankar, S., Zamfirescu-Pereira, J. D., Hartmann, B., Parameswaran, A. G., & Arawjo, I. (2024). Who validates the validators? Aligning LLM-assisted evaluation of LLM outputs with human preferences. In *Proceedings of the 37th Annual ACM Symposium on User Interface Software and Technology*. Association for Computing Machinery. https://doi.org/10.1145/3654777.3676450
  Scope: introduces EvalGen and the criteria-drift problem, supporting human alignment, held-out validation and explicit validator provenance.
- Pradeep, R., Thakur, N., Upadhyay, S., Campos, D., Craswell, N., & Lin, J. (2025). *The great nugget recall: Automating fact extraction and RAG evaluation with large language models* (arXiv preprint arXiv:2504.15068). https://arxiv.org/abs/2504.15068
  Scope: reports AutoNuggetizer variants calibrated against human-based conditions and explicitly notes remaining per-topic diagnostic limitations.
- Norgaila, E., Daniela, L., & Kalniņa, D. (2026). Reflective prompt engineering for assessment rubric optimization: An empirical study of human–AI alignment. *Technology, Knowledge and Learning, 31*, 1023–1038. https://doi.org/10.1007/s10758-026-09979-2
  Scope: provides a recent rubric-refinement and human–AI alignment study; it is evidence for an experiment, not a release-time validity guarantee or a substitute for psychometric calibration.

## 6. Automated essay scoring / automated scoring as many-facet measurement — Accepted baseline / evolving

Architecture effect:

- scorer output is governed observation evidence, not an oracle score;
- human and AI raters share a calibration framework;
- severity, agreement, range use, DIF/fairness, drift and adjudication are separate evidence dimensions;
- raw human-AI correlation is insufficient as the primary validity claim.

Primary basis:

- Williamson, D. M., Xi, X., & Breyer, F. J. (2012). A framework for evaluation and use of automated scoring. *Educational Measurement: Issues and Practice, 31*(1), 2–13.
- Uto, M., & Ueno, M. (2020). A generalized many-facet Rasch model and its Bayesian estimation using Hamiltonian Monte Carlo. *Behaviormetrika, 47*, 469–496.
- American Educational Research Association, American Psychological Association, & National Council on Measurement in Education. (2014). *Standards for educational and psychological testing*. American Educational Research Association.

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
- recover crossed / multiple-membership `u_h` with Rust MAP + RMSE evidence;
- keep OLS/AR and continuous-time claims on separate estimator slices.

Primary basis:

- Fox, J.-P., & Glas, C. A. W. (2001). Bayesian estimation of a multilevel IRT model. *Psychometrika, 66*, 271–288. https://doi.org/10.1007/BF02294839
- Uto, M. (2023). A Bayesian many-facet Rasch model with Markov modeling for rater severity drift. *Behavior Research Methods, 55*, 3910–3928. https://doi.org/10.3758/s13428-022-01997-z
- Jeon et al. (2021) and Kang and Jeon (2025) for residual latent-space interaction after explicit hierarchy/time, not as a substitute for those structures.
- American Educational Research Association, American Psychological Association, & National Council on Measurement in Education. (2014). *Standards for educational and psychological testing*. American Educational Research Association.
- Browne, W. J., Goldstein, H., & Rasbash, J. (2001). Multiple membership multiple classification (MMMC) models. *Statistical Modelling, 1*(2), 103–124.

The first Rust state layer (ADR-0019) is an independent per-respondent OLS
trend and a caller-supplied discrete AR predictor. It is not the Fox and Glas
multilevel IRT estimand and does not estimate population random effects.

ADR-0020 is a separate joint MAP hierarchical continuous-time AR(1) Rasch
slice. It estimates shared `(mu, tau, lambda)` and person-occasion states
with Wald observed-information intervals. It is not Fox and Glas Gibbs
sampling, not Jeon and Rabe-Hesketh adaptive-quadrature ML, and not estimated
multiple-membership `u_h`.

- Jeon, M., & Rabe-Hesketh, S. (2016). An autoregressive growth model for longitudinal item analysis. *Psychometrika, 81*(3), 830–850. https://doi.org/10.1007/s11336-015-9489-2
- Laird, N. M., & Ware, J. H. (1982). Random-effects models for longitudinal data. *Biometrics, 38*(4), 963–974. https://doi.org/10.2307/2529876
- Oravecz, Z., Tuerlinckx, F., & Vandekerckhove, J. (2011). A hierarchical latent stochastic differential equation model for affective dynamics. *Psychological Methods, 16*(2), 468–490. https://doi.org/10.1037/a0024375

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

The complete APA 7 records and evidence classification for Conductor, TRINITY
and the vendor-described Fugu system are maintained in
[`docs/doctoring/llm_orchestration_test_time_compute.md`](../doctoring/llm_orchestration_test_time_compute.md).
That record distinguishes peer-reviewed/preprint research from vendor product
evidence and keeps the implementation claim bounded to an experiment plan.

## 12. Standards baseline

Architecture/product documentation also uses:

- ISO/IEC/IEEE 29148:2018 for requirements engineering;
- ISO/IEC/IEEE 42010:2022 for architecture-description concerns/viewpoints;
- ISO/IEC 25010:2023 for product-quality concerns;
- ISO/IEC 42001:2023 for AI lifecycle/governance concerns where applicable;
- WCAG 2.2 for accessible report surfaces;
- AERA/APA/NCME 2014 Testing Standards for evidence, fairness and score-use interpretation.
- National Institute of Standards and Technology. (2023). *Artificial intelligence risk management framework (AI RMF 1.0)* (NIST AI 100-1). https://doi.org/10.6028/NIST.AI.100-1
  Governance scope: Govern, Map, Measure and Manage are used as a risk-management input; this citation is not a certification or conformity claim.
- National Institute of Standards and Technology. (2024). *Artificial intelligence risk management framework: Generative artificial intelligence profile* (NIST AI 600-1). https://doi.org/10.6028/NIST.AI.600-1
  Governance scope: provider/model risk, provenance, evaluation, human oversight and incident considerations; this citation is not a certification or conformity claim.

## 13. Angoff delta-plot observed-score DIF — Accepted screen

Architecture effect:

- keep Angoff's transformed-item-difficulty / delta-plot as a named small-sample observed-score DIF screen;
- keep Mantel–Haenszel, logistic DIF, and SIBTEST as distinct matching-score / standardization screens;
- treat a flag as a review candidate, not a fairness determination;
- do not cite CWE, OWASP, or NIST as the method basis.

Primary basis:

- Angoff, W. H. (1972, September). *A technique for the investigation of cultural differences* [Paper presentation]. Annual meeting of the American Psychological Association, Honolulu, HI, United States.
- Angoff, W. H., & Ford, S. F. (1973). Item-race interaction on a test of scholastic aptitude. *Journal of Educational Measurement, 10*(2), 95–105. https://doi.org/10.1111/j.1745-3984.1973.tb00787.x
- Magis, D., & Facon, B. (2012). Angoff's Delta method revisited: Improving DIF detection under small samples. *British Journal of Mathematical and Statistical Psychology, 65*(2), 302–321. https://doi.org/10.1111/j.2044-8317.2011.02025.x
- Magis, D., & Facon, B. (2014). deltaPlotR: An R package for differential item functioning analysis with Angoff's Delta Plot. *Journal of Statistical Software, 59*(1), 1–19. https://doi.org/10.18637/jss.v059.c01
- American Educational Research Association, American Psychological Association, & National Council on Measurement in Education. (2014). *Standards for educational and psychological testing*. American Educational Research Association.

Decision record: ADR-0018. Method page: `docs/delta_plot_dif.md`. Adjacent MH/logistic/SIBTEST sources remain in `docs/rubric_dif_pilot_handoff.md`.

## 14. Bradley–Terry MM pairwise ranking — Accepted estimators

Architecture effect:

- fit tie-free paired comparisons with Bradley–Terry worths by Hunter MM;
- fit observed ties with the implemented additive-`alpha0` BRATT variant only;
- do not claim Rao–Kupper or Davidson unless a later complete model path implements them;
- keep LSR / I-LSR / Plackett–Luce as separate ranking estimators.

Primary basis:

- Bradley, R. A., & Terry, M. E. (1952). Rank analysis of incomplete block designs: I. The method of paired comparisons. *Biometrika, 39*(3/4), 324–345. https://doi.org/10.2307/2334029
- Hunter, D. R. (2004). MM algorithms for generalized Bradley–Terry models. *The Annals of Statistics, 32*(1), 384–406. https://doi.org/10.1214/aos/1079120141
- American Educational Research Association, American Psychological Association, & National Council on Measurement in Education. (2014). *Standards for educational and psychological testing*. American Educational Research Association.

Decision record: ADR-0017. Method page: `docs/bradley_terry_mm.md`. Changelog notes that Bradley and Terry (1952) or Hunter (2004) were unread at port time are historical source-governance comments, not a reason to omit these citations.

## Maintenance rule

When a new research finding changes a released formula, score interpretation, factor/model relation, validity gate, item/rubric lifecycle or rater-evaluation rule, update the relevant ADR, requirement IDs and this traceability record in the same reviewed change.
