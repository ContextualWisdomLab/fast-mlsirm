# Corpus triage — batch 3 (~100 papers)

Disposition of the third supplied reading set. "This batch" = implemented in
the current change set; "covered" = the capability already exists (with the
earlier citation basis); "roadmap" = requires a capability the binary
marginal engine does not have yet (reason given); "foundational/context" =
reviews, applications, or textbooks that inform documentation, not code.

## Implemented in this batch

| Paper | Feature |
|---|---|
| Magis (2013), item information of the 4PL | closed-form 4PL item information (`item_information_4pl`, reduces to the 2PL at `c=0, d=1`); item/test information surfaces in the scoring module |
| Bock & Mislevy (1982), adaptive EAP estimation | sequential EAP scoring + maximum-information CAT item selection over a frozen bank |
| Wang, Kuo & Chao (2010), MCAT system | the CAT loop generalized to the multidimensional simple-structure bank (per-dimension information targeting) |
| Marsman et al. (2016), plausible values | posterior plausible-value draws from the scoring grid (secondary-analysis exports) |
| Guo, Zheng & Chang (2015), stepwise TCC drift | test-characteristic-curve drift detection between two calibrations of a common bank (stepwise anchor purification) |
| Haberman, Sinharay & Chon (2013), residual item fit | standardized residuals of observed vs estimated ICCs on the score grid |
| Sinharay (2016), resampling person fit | fixed-EAP conditional Monte Carlo approximation for `l_z*` (add-one-smoothed empirical lower-tail frequencies); the complete generalized resampling procedure, including replicate-wise ability re-estimation, is not implemented |
| Tay & Drasgow (2012), adjusted chi2/df | Exploratory repository-specific item-pair ratios only. The paper finds a fixed cutoff insufficient and recommends a parametric bootstrap; that inferential procedure is not implemented. |

## Already covered (earlier basis)

- Bock & Lieberman (1970); Bock & Mislevy EAP; Meng & Schilling (1996);
  Drasgow, Levine & Williams (1985); Tay et al. (2011); Ames & Penfield
  (2015, NCME item-fit module); Sueiro & Abad (2011, nonparametric fit
  context); Sinharay (2006, Bayesian item fit → PPMC documented-only);
  Sinharay (2015, mixed-format person fit → binary case covered): the
  marginal EM, EAP/EAPsum, S-X2, l_z/l_z* stack.
- Jeon & Rijmen (2016, flirt modularity); Chalmers (2015 MH-RM mixed-effects;
  2016 mirtCAT): engine-design references — the modular estimator axes
  (population, anchors, covariate, mixture) mirror flirt's design; MH-RM
  remains the documented alternative engine.
- Williamson-framework relatives (Bennett 1991/2011; Martinez & Bennett 1992;
  Ramineni & Williamson 2013; Higgins et al. 2011; Zechner et al. 2015; Liu
  et al. 2014; Advancing Human Assessment 2017): the `validate_judge` gate
  set is the operational core; these inform its documentation and thresholds.
- Wilson, De Boeck & Carstensen (2008, explanatory IRT): person-side
  explanation = multigroup/multilevel structures; item-side (LLTM) is on the
  roadmap below.
- Lord (1974, omitted responses); Kadengye et al. (2014); Bolsinova & Maris
  (2016): MAR handling by direct marginal likelihood is the implemented
  position; not-reached/omit distinctions documented.
- Finch & Pierson (2011, mixture IRT): the zero-inflated mixture is the first
  instance; general C-class mixtures on the roadmap.
- Ferrando (2016, person discrimination/fluctuation): person-fit family
  covers the diagnostic use; the person-discrimination parameter itself is a
  model extension (roadmap, low priority).

## Roadmap (capability gaps, with reasons)

- **Polytomous responses** — Muraki (1990) GPCM; Penfield (2014); Muraki &
  Carlson (1995); Dodd et al. (1995 CAT); Kang & Chen-style S-X2
  generalizations; Likert/two-decision (Thissen-Roe & Thissen 2013); ERS
  models (Jin & Wang 2014); rating-scale distances (2004): the engine is
  binary-only; a categorical-kernel model family is a separate model-design
  PR.
- **3PL/4PL response functions** — Barton & Lord (1981): estimating lower/
  upper asymptotes changes the response kernel across the table builder and
  every M-step; the information side (Magis) landed first, the estimation
  side is the next model-design PR (the table architecture keeps the GPU
  kernels untouched).
- **Bifactor family (`BIFAC2PLM`)** — Gibbons & Hedeker (1992), Cai, Yang &
  Hansen (2011): the general factor slots into the engine's conditional-
  factorization E-step exactly like the latent-space coordinate (the
  Gibbons-Hedeker dimension reduction), so this is the highest-leverage next
  model variant; then testlet models (Li, Li & Wang
  2010 polytomous testlets; Paek & Fukuhara 2015 testlet DIF), projective
  equating (Kim & Cho 2019), vertical scaling with construct shift (Li &
  Lissitz 2012), bifactor MCAT design (Seo & Weiss 2015), bifactor
  dimensionality suites (Immekus & Imbrie 2008; Reise, Moore & Haviland
  2010): follow-ons once BIFAC2PLM stabilizes.
- **Cognitive diagnosis models** — DINA invariance (de la Torre & Lee 2010),
  CDM framework (de la Torre & Minchen 2014), Wald DIF in CDM (Hou et al.
  2014), scaling hybrid (Bradshaw & Templin 2014), Wilson (2008): different
  measurement paradigm (attribute mastery), out of the latent-trait engine's
  scope.
- **MCMC/Bayesian estimation** — Patz & Junker (1999), Natesan et al. (2016),
  Martin-Fernandez & Revuelta (2017), Revuelta & Ximenez (2017), Sinharay
  (2006 PPMC), Fox et al. (2014 randomized-response MIRT): AGENTS.md scopes
  this package as intentionally not a Bayesian sampler; deterministic
  EM/QMC-EM is the estimation contract.
- **Limited-information overall fit (M2/RMSEA2)** — Maydeu-Olivares & Joe
  (2014), Cai-style adjustments: valuable; needs bivariate-margin delta
  matrices and weighted chi-square tails — planned next after this batch.
- **Equating/linking beyond FIPC** — Ryan & Brockmann (2011 primer), Kim &
  Lee (2006 mixed-format linking), Ali & van Rijn (2015 parallel-forms
  targets — partially served by `assemble_test_form`), Bolsinova & Maris
  (2016): concurrent calibration + FIPC cover the operational need here;
  moment-based linking transformations (mean-mean/Stocking-Lord) are a small
  future utility.
- **Item-side explanatory structure (LLTM)** — Wilson et al. (2008), Park &
  Liu (2019), Embretson & Yang (2013 multicomponent): difficulty design
  matrices `b = W delta`; natural extension of the covariate machinery.
- **Specialized response processes** — ideal-point models (Maydeu-Olivares
  et al. 2006 GGUM-family; Carter & Dalal 2010; Chernyshenko 2007;
  Tay 2011 covered as fit-comparison), forced choice (Hontangas et al. 2015),
  response certainty (Ferrando et al. 2013), diffusion-IRT (van der Maas et
  al. 2011), response-time effort (Wise & DeMars 2006), fMRI application
  (Thomas et al. 2013), non-compensatory calibration (Wang & Nydick 2015):
  distinct kernels; the latent-space distance term already gives one
  ideal-point-like mechanism (documented relation).
- **Flexible/nonparametric ICCs** — Liang & Browne (2015 quasi-parametric),
  Camilli & Fox (2015 aggregate EFA), Zhang (2013 dimensionality across
  designs), Ip et al. (2013 functional unidimensionality): exploratory-side
  tools; Q3/GDDM + dimensionality_diagnostics are the current instruments.

## Foundational / context (documentation only)

Lord (1980, Applications of IRT); Rabe-Hesketh, Skrondal & Pickles (2004
GLLAMM — the general framework our multilevel structure instantiates);
Parsons & Hulin (1982); Schmitt, Cortina & Whitney (1993); Reise & Flannery
(1996); Edelen & Reeve (2007); Rusch et al. (2017); Christensen et al.
(2016); Terluin et al. (2018); Segall (2001); Ogasawara (2010); Manrique-
Vallier et al. (2014, structural zeros in categorical imputation — cf. the
ZI mixture); Sliter & Zickar (2014); Chernyshenko et al. (2001); remaining
application/review PDFs of the batch.
