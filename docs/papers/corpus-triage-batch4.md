# Corpus triage — batch 4 (~75 papers)

Disposition of the fourth supplied reading set (same legend as batch 3).

## Implemented in this batch

| Paper | Feature |
|---|---|
| Chen & Thissen (1997), local-dependence indexes for item pairs | `ld_indices()` — signed standardized pairwise LD X2 and G2 against the model-implied joint probabilities |

## Already covered (earlier basis)

- Orlando, Thissen & Thissen (2000): S-X² — implemented (batch 1 of this
  branch). Kang & Chen (2008, 2011): its polytomous generalizations — the
  binary case is implemented; polytomous with the response-kernel roadmap.
- Thissen, Pommerich, Billeaud & Williams (1995): EAPsum — implemented
  (scoring.rs, conversion tables in the serving bundle). Cai (2015)
  Lord-Wingersky 2.0: the recursion is implemented on the joint grid; the
  hierarchical (bifactor) version lands with `BIFAC2PLM`.
- Cai (2010), Cai & Angeles (2015), Chalmers & Flora (2014 MH-RM): MH-RM
  remains the documented alternative engine (AGENTS.md scopes the package to
  deterministic EM; QMC-EM covers the high-dimensional-integral need).
- Wang (2010) IRT-ZIP: the structural-zero mixture is implemented for the
  binary kernel; the count-response (Poisson) kernel is a response-kernel
  roadmap item.
- Yuan, Cheng & Patton (2014): information-matrix SE comparison — the Oakes
  observed-information estimator is implemented; the sandwich/XPD variants
  are a small follow-on once misspecification-robust SEs are needed.
- Chalmers, Counsell & Flora (2016) DIF effect sizes: `dif_analysis` reports
  logit-scale effect sizes; their DRF/sDRF integrals are a natural extension
  of the same virtual-item machinery.
- Meade & Craig (2012) careless responding: the person-fit stack (l_z*,
  resampling p-values, screening weights) is the operational instrument;
  their survey-specific indices (longstring, even-odd) are preprocessing
  utilities outside the model core.
- Liu & Maydeu-Olivares (2013, 2014): local-dependence diagnostics land with
  `ld_indices`/Q3/GDDM/adjusted chi2; the source-of-misfit decomposition
  belongs to the M2 roadmap item.
- Lord (1986 MLE/Bayes estimation), Bock & Mislevy EAP, Reckase (2009),
  Carlson (1988), Wirth & Edwards (2007), Cai, Choi & Kuhfeld (2016 IRT
  overview): estimation/scoring foundations of the implemented engine.
- Sulis & Toland (2017 multilevel IRT intro): the implemented multilevel
  structure; Höhler et al. (2010) within/between multidimensionality:
  covered by simple-structure multidim + multilevel intercepts.
- Makransky, Mortensen & Glas (2013 MCAT): `cat_next_item` implements the
  multidimensional adaptive loop for the binary bank.

## Roadmap (consolidated across batches; explicitly requested)

Priority order for the next model-design PRs on this branch's foundation:

1. **`BIFAC2PLM` bifactor family** — Gibbons & Hedeker (1992); Cai, Yang &
   Hansen (2011); Cai & Hansen (2013); Cai (2015 LW 2.0); Toland et al.
   (2017); Li & Rupp (2011 S-X² under bifactor); Liu & Thissen (2012 score
   test); Huang et al. (2013 higher-order traits): the general factor enters
   the existing conditional-factorization E-step exactly like the
   latent-space coordinate (inner-product term `lambda_i * g`), so the
   engine structure carries over; higher-order models follow as constrained
   bifactor.
2. **Limited-information overall fit (M2, RMSEA2)** — Maydeu-Olivares & Joe
   (2005, 2014); Maydeu-Olivares (2013); Cai & Hansen (2013); Hansen et al.
   (2016); Liu & Maydeu-Olivares (2014): univariate+bivariate margins,
   delta-matrix reduction, weighted chi-square tail.
3. **3PL/4PL estimation** — Barton & Lord (1981): response-kernel change;
   the information side (Magis 2013) already landed.
4. **Polytomous response kernels** — Muraki (1990, 1993 GPCM + information);
   Muraki & Carlson (1995); Falk & Cai (2016 monotonic-polynomial GPCM);
   Thissen et al. (1995 already covers the scoring recursion); Kang & Chen
   (2008, 2011); Emons (2008 polytomous person fit); Jiao & Zhang (2015
   polytomous multilevel testlets); response-style models (van Rosmalen et
   al. 2010; Huang 2016 mixture random-effect ERS; Kam & Fan 2018; Wang et
   al. 2014 wording effects): unlocks the largest cluster of remaining
   papers; design decision — categorical kernel with per-category tables.
5. **Linking/equating utilities** — Yao & Boughton (2009 multidimensional
   linking); Brossman & Lee (2013 MIRT observed/true-score equating); Chen
   et al. (2009 common-scale linking): moment/characteristic-curve
   transformations over the existing bank structures.
6. **Response-time integration** — van der Linden, Klein Entink & Fox
   (2010 collateral information); Veldkamp (2016 RT in CAT); Partchev & De
   Boeck (2013 power/speed): a lognormal RT sidecar likelihood sharing the
   person posterior.
7. **Robust/misspecification tooling** — Bolt, Deng & Lee (2014 vertical-
   scaling misspecification); Bonifay & Cai (2017 model complexity); Hooker,
   Finkelman & Schwartzman (2009 paradoxical MIRT scoring — a documentation
   caveat for multidimensional score reporting); Wang (2015 latent-trait
   estimation properties).

## Foundational / context (documentation only)

Reise & Revicki (Handbook of Item Response Theory Modeling); van der Linden
(Handbook of IRT); Maydeu-Olivares (2013 GOF overview — the M2 roadmap's
frame); Holland (1990 sampling foundations); Wirth & Edwards (2007);
clinical/applied papers (Reise & Waller 2009; Reise & Haviland 2005; Velozo
et al. 2012; Chen et al. 2009 PROMIS-style linking application; Terluin et
al.; Martínez-Plumed et al. 2016 IRT-in-ML; Higgins & Heilman 2014 gaming
susceptibility — operational guidance for the judge-validation gates);
survey-nonresponse context (Shoemaker et al. 2002; Frick & Grabka 2010;
Si-Reiter nonparametric Bayesian imputation) — the MAR-marginalization
position plus the ZI mixture cover the modeling side.
